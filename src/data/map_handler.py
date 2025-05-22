# src/data/map_handler.py

import os
import pickle
from typing import Any, List, Dict

import networkx as nx
import osmnx as ox


class RegionMap:
    """
    Container for a map region: waypoints and the raw road graph.
    """
    def __init__(self, waypoints: List[Dict[str, float]], road_graph: nx.Graph):
        self.waypoints = waypoints
        self.road_graph = road_graph


class MapHandler:
    """
    Handles fetching and caching OpenStreetMap regions.
    Expects region_id strings of the form "lat_min,lon_min,lat_max,lon_max".
    Caches each region as a pickle in cache_dir.
    """

    def __init__(self, cache_dir: str, network_type: str = "drive"):
        """
        Args:
            cache_dir: Directory to store cached RegionMap pickles.
            network_type: OSMnx network type, e.g. "drive", "walk", etc.
        """
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.network_type = network_type

    def get_region(self, region_id: str) -> RegionMap:
        """
        Load a region from cache or fetch from OSM if missing.

        Args:
            region_id: Bounding box encoded as "lat_min,lon_min,lat_max,lon_max"
        Returns:
            RegionMap with .waypoints (list of {"lat", "lon"}) and .road_graph (networkx.Graph)
        """
        cache_path = os.path.join(self.cache_dir, f"{region_id.replace(',', '_')}.pkl")
        # Return from cache if available
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        # Parse bounding box
        try:
            lat_min, lon_min, lat_max, lon_max = map(float, region_id.split(","))
        except ValueError:
            raise ValueError(
                f"Invalid region_id '{region_id}'. Expected 'lat_min,lon_min,lat_max,lon_max'."
            )

        # Fetch road network from OSM
        graph = ox.graph_from_bbox(
            north=lat_max,
            south=lat_min,
            east=lon_max,
            west=lon_min,
            network_type=self.network_type,
        )

        # Extract node coordinates as waypoints
        waypoints = [
            {"lat": data["y"], "lon": data["x"]}
            for _, data in graph.nodes(data=True)
        ]

        region_map = RegionMap(waypoints=waypoints, road_graph=graph)

        # Cache to disk
        with open(cache_path, "wb") as f:
            pickle.dump(region_map, f)

        return region_map
