from datetime import timedelta
import numpy as np
import pytest
from test_review_1111 import field
from test_hydrobrain import NOW, _wx
from hydrobrain.decision import infiltration_plan
from hydrobrain.ml_forecast import MLCore


def test_near_equal_rate_does_not_overdose():
    p = infiltration_plan(55.3, 20, 19.8)
    assert p['duration_min'] * 20 / 60 == pytest.approx(55.3, abs=.001)


def test_daily_plan_bounded_even_at_low_ksat():
    p = infiltration_plan(30, 20, .1)
    assert p['total_h'] <= 24 and p['deliver_exact'] <= 2.4 + 1e-8
    assert p['limited']


def test_cycle_water_spans_hours_and_survives_restart(tmp_path):
    b, fid = field(tmp_path, ksat_mm_h=2, application_rate_mm_h=20, irrigation_method='sprinkler', forecast_core='physics')
    b.recommend(fid, now=NOW, weather=_wx([]))
    b.log_irrigation(fid, 20, time=NOW, cycles=True)
    b.recommend(fid, now=NOW + timedelta(hours=1), weather=_wx([]))
    assert not b.load_field(fid)['events']['irrigation'][0].get('applied')
    b.recommend(fid, now=NOW + timedelta(hours=10), weather=_wx([]))
    assert b.load_field(fid)['events']['irrigation'][0]['applied']


def test_legacy_local_model_is_rejected(tmp_path):
    import joblib
    from pathlib import Path
    b, fid = field(tmp_path)
    f=b.load_field(fid); f['ml_local']={'enabled':True}
    path=Path(b._local_path(fid)); path.parent.mkdir(exist_ok=True)
    joblib.dump({1:'legacy'},path)
    assert b._load_local(f)=={}


def test_stage4_rejects_production_model(monkeypatch):
    monkeypatch.delenv('HYDRO_EVALUATION_MODEL',raising=False)
    with pytest.raises(ValueError,match='independently'):
        MLCore.load_for_evaluation()


def test_calibration_is_opt_in(tmp_path):
    b,fid=field(tmp_path)
    assert b.load_field(fid)['params']['pstress_calibration'] is False
