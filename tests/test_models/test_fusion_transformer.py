import pytest
from src.data.map_handler import MapHandler, RegionMap


def test_invalid_region_id_format(tmp_path):
    mh = MapHandler(cache_dir=str(tmp_path))
    with pytest.raises(ValueError):
        mh.get_region("invalid_format")

# Note: Further tests for map_handler require OSMnx network fetch and cache behavior,
# which may depend on external internet connectivity."