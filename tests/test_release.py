"""Scientific regressions and reproducibility checks for the manuscript release."""

import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from py_sz_model.config import DATA_DIR, FROZEN_MODELS_DIR, SZ_FEATURE_NAMES, load_sz_training_data
from py_sz_model.manuscript import build, check_release, interval_summary, load_reconstructions, moving_average, test_set_metrics
from py_sz_model.models import ModelData, TrainingModel
from py_sz_model.prediction import run_predictions
from py_sz_model.r_calculation import calculate_R_for_dataframe, calculate_R_with_uncertainty, parse_args
from py_sz_model.ratios import calculate_ratios
from py_sz_model.training import train_sz_models
from py_sz_model.randomness import MC_RANDOM_SEED


class ManuscriptReleaseTests(unittest.TestCase):
    def test_frozen_membership_models_and_uncertainties(self):
        self.assertEqual(check_release()['manuscript_samples'], 92)

    def test_build_reproduces_paper_and_archived_proxy_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            result = build(Path(directory))
        self.assertEqual(result['stage_n'], [92, 30, 27, 35])
        np.testing.assert_allclose(result['stage_means_ppm'][1:], [309.514856528,235.365944595,325.938998186], atol=1e-10)
        self.assertAlmostEqual(result['drawdown_ppm'], 119.794176070, places=9)
        self.assertEqual(result['figure2c']['within_combined_1sigma'],20)
        self.assertEqual(result['proxy_checks']['prediction_rows'],1337)

    def test_test25_co2_agreement_matches_manuscript(self):
        metrics = test_set_metrics()
        gb = metrics[(metrics.model=='GradientBoosting') & (metrics.target=='CO2')].iloc[0]
        self.assertEqual(gb['n'],25)
        self.assertEqual(gb.within_prediction_1sigma,19)
        self.assertEqual(round(gb.mean_residual_ppm),10)
        self.assertEqual(round(gb.residual_sample_SD_ppm),41)

    def test_stage_boundaries_do_not_double_count(self):
        synthetic = pd.DataFrame({'Age (Ma)':[4.,5.2,5.3,5.9,6.,6.5],
            'site':['Jiaxian']*6,'CO2_mean':[100.]*6,'CO2_std':[10.]*6,
            'R':[.5]*6,'R_std':[.02]*6,'Sz_mean':[200.]*6,'Sz_std':[8.]*6})
        sizes = [interval_summary(synthetic,'a',4,5.3)['n'],
                 interval_summary(synthetic,'b',5.3,6)['n'],
                 interval_summary(synthetic,'c',6,6.5,True)['n']]
        self.assertEqual(sizes,[2,2,2])

    def test_moving_average_combines_two_distinct_variances(self):
        data = pd.DataFrame({'Age (Ma)':[4.,4.1], 'CO2_mean':[100.,200.], 'CO2_std':[10.,20.]})
        result = moving_average(data)
        np.testing.assert_allclose(result.mean_CO2_ppm,150.)
        # Variance of measurement mean = (100+400)/4; sample-mean variance = 5000/2.
        np.testing.assert_allclose(result.combined_SE_ppm,np.sqrt(2625.))

    def test_missing_required_input_is_not_silently_removed(self):
        data = pd.DataFrame({'x':[1.,np.nan]})
        with self.assertRaisesRegex(ValueError,'no samples are silently dropped'):
            ModelData(data,['x'])


class MonteCarloTests(unittest.TestCase):
    def test_all_default_seed_entrypoints_use_42(self):
        from py_sz_model import prediction, training, ratios
        from py_sz_model.config import DATA_RANDOM_STATE, MODEL_RANDOM_STATE
        self.assertEqual((MC_RANDOM_SEED, DATA_RANDOM_STATE, MODEL_RANDOM_STATE), (42,42,42))
        for parser, argv in [(prediction.parse_args,['sz-predict']),
                             (training.parse_args,['sz-train']),
                             (parse_args,['sz-calc-r','sample.csv'])]:
            with patch('sys.argv',argv):
                self.assertEqual(parser().seed,42)
        self.assertEqual(inspect.signature(ratios.calculate_ratios).parameters['random_state'].default,42)

    def test_default_r_protocol_is_zero_correction_and_100000_draws(self):
        for function in (calculate_R_with_uncertainty,calculate_R_for_dataframe):
            parameters=inspect.signature(function).parameters
            self.assertEqual(parameters['num_simulations'].default,100000)
            self.assertEqual(parameters['decomp_corr_mean'].default,0.)
            self.assertEqual(parameters['decomp_corr_std'].default,0.)
        with patch('sys.argv',['sz-calc-r','sample.csv']):
            args=parse_args()
        self.assertEqual((args.num_simulations,args.decomp_corr_mean,args.decomp_corr_std),(100000,0.,0.))

    def test_r_seed_repeats_without_restarting_each_row_or_global_rng(self):
        row=pd.read_csv(DATA_DIR/'prediction_set/Shilou_features_bulk.CSV').iloc[[0]]
        data=pd.concat([row,row],ignore_index=True)
        global_state=np.random.get_state()
        a=calculate_R_for_dataframe(data,num_simulations=200,random_state=37)
        b=calculate_R_for_dataframe(data,num_simulations=200,random_state=37)
        pd.testing.assert_frame_equal(a,b)
        self.assertNotEqual(a.R.iloc[0],a.R.iloc[1])
        now=np.random.get_state()
        np.testing.assert_array_equal(global_state[1],now[1])
        self.assertEqual(global_state[2:],now[2:])

    def test_ratio_protocols_have_correct_limits_and_seed(self):
        data=pd.DataFrame({'aFe':[2.],'aSi':[4.],'aAl':[1.],'fFe':[5.],
                           'aFe_std':[.2],'aSi_std':[.4],'aAl_std':[0.],'fFe_std':[0.]})
        direct=calculate_ratios(data,'first-order')
        self.assertAlmostEqual(direct['aFe/aSi'].iloc[0],.5)
        self.assertAlmostEqual(direct['aFe/aSi_std'].iloc[0],np.sqrt(.005))
        a=calculate_ratios(data,'monte-carlo',random_state=37)
        b=calculate_ratios(data,'monte-carlo',random_state=37)
        pd.testing.assert_frame_equal(a,b)
        self.assertAlmostEqual(a['aAl/fFe_std'].iloc[0],0.,places=14)

    def test_training_augmentation_keeps_74_25_membership_and_is_seeded(self):
        a,b=load_sz_training_data(),load_sz_training_data()
        a.pre_process(test_size=.25,n_mc_samples=10,mc_random_state=37)
        b.pre_process(test_size=.25,n_mc_samples=10,mc_random_state=37)
        self.assertEqual((len(a.train_set),len(a.test_set),len(a.features_train_mc)),(74,25,740))
        self.assertFalse(set(a.train_set.Sample_ID)&set(a.test_set.Sample_ID))
        pd.testing.assert_frame_equal(a.features_train_mc,b.features_train_mc)
        pd.testing.assert_series_equal(a.target_train_mc,b.target_train_mc)

    def test_zero_feature_uncertainty_returns_fixed_model_predictions(self):
        q=load_sz_training_data().data.iloc[:3]
        wrapper=TrainingModel('GradientBoosting')
        wrapper.load_model(FROZEN_MODELS_DIR/'GradientBoosting_sz.joblib')
        x=q[SZ_FEATURE_NAMES]
        zero_sd=pd.DataFrame(np.zeros(x.shape),columns=x.columns,index=x.index)
        predictions,uncertainties=wrapper.predict_with_uncertainty(x,zero_sd,8,random_state=37)
        np.testing.assert_allclose(predictions,wrapper.predict(x),atol=1e-10)
        np.testing.assert_allclose(uncertainties,0.,atol=1e-10)

    def test_seeded_frozen_model_prediction_is_repeatable_for_92_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            a=run_predictions(output_dir=base/'a',n_mc_samples=12,random_state=37)
            b=run_predictions(output_dir=base/'b',n_mc_samples=12,random_state=37)
            self.assertEqual(len(a),2)
            self.assertEqual(sum(len(pd.read_csv(p)) for p in a),92)
            for x,y in zip(a,b):
                self.assertEqual(x.read_bytes(),y.read_bytes())

    def test_recalculating_r_does_not_change_sz_draws(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            a=run_predictions(output_dir=base/'cached',n_mc_samples=8,random_state=42)
            b=run_predictions(output_dir=base/'new_r',n_mc_samples=8,r_simulations=100,recalculate_r=True,random_state=42)
            for x,y in zip(a,b):
                pd.testing.assert_frame_equal(pd.read_csv(x)[['Sz_mean','Sz_std']],pd.read_csv(y)[['Sz_mean','Sz_std']])

    def test_ridge_training_cli_path_is_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for variant in ['a','b']:
                train_sz_models(model_names=['Ridge'],models_dir=root/variant/'models',
                    metrics_dir=root/variant/'metrics',predictions_dir=root/variant/'predictions',
                    n_mc_samples=10,n_jobs=1,seed=37)
            for rel in ['models/training_run.json','metrics/Ridge_sz.csv','predictions/Ridge_sz.csv']:
                self.assertEqual((root/'a'/rel).read_bytes(),(root/'b'/rel).read_bytes())
            run=json.loads((root/'a/models/training_run.json').read_text())
            self.assertEqual((len(run['train_ids']),len(run['test_ids'])),(74,25))


if __name__ == '__main__':
    unittest.main()
