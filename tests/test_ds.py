"""Перевірки допоміжного коду DS-частини (без мережі)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pd = pytest.importorskip("pandas")

from ds import ismn  # noqa: E402

STM = """FR_Aqui FR_Aqui Fraye 44.46700 -0.72690 55.00 0.05 0.05 ThetaProbe-ML2X
2016/05/01 00:00 0.2500 G M
2016/05/01 01:00 0.2480 G M
2016/05/01 02:00 0.9000 D02 M
2016/05/01 03:00 0.2460 G M
"""


def test_ismn_stm_parsing_and_flags(tmp_path):
    d = tmp_path / "FR_Aqui" / "Fraye"
    d.mkdir(parents=True)
    (d / "FR_Aqui_FR_Aqui_Fraye_sm_0.050000_0.050000_ThetaProbe-ML2X_20160501_20160501.stm").write_text(STM)
    (d / "FR_Aqui_FR_Aqui_Fraye_ts_0.050000_0.050000_ThetaProbe-ML2X_20160501_20160501.stm").write_text(STM)
    raw = ismn.read_stm(next(d.glob("*_sm_*.stm")))
    assert len(raw) == 4 and raw.attrs["station"] == "Fraye" and raw.attrs["depth_to"] == 0.05
    st = ismn.load_station(tmp_path, station="Fraye", depth_max=0.10)
    assert len(st) == 3 and st["sm"].max() < 0.3          # значення з прапорцем D02 відкинуто, ts-файл ігноровано
    with pytest.raises(ValueError):
        ismn.load_station(tmp_path, station="Fraye", depth_max=0.01)


def test_ismn_station_name_is_exact(tmp_path):
    for st in ("hillan", "hillan2"):
        d = tmp_path / "FR_Aqui" / st
        d.mkdir(parents=True)
        (d / f"FR-Aqui_FR-Aqui_{st}_sm_0.050000_0.050000_X_20160501_20160501.stm").write_text(STM)
    assert len(ismn.find_files(tmp_path, "hillan")) == 1
    assert len(ismn.find_files(tmp_path, "hillan2")) == 1
