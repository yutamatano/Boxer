"""Read explicit pose provenance; legacy Neon geometry remains metric Z-up."""

import json


def _verified_colmap_world(meta):
    return (
        meta.get("metric") is True
        and meta.get("pose_units") == "metres"
        and meta.get("scale_status") == "independently_verified"
        and meta.get("world_up_status") == "gravity_aligned"
        and meta.get("quality_status") == "pass"
    )


def geometry_metadata(data):
    if "geometry_metadata" not in data:
        # A COLMAP/visual archive must never acquire legacy metric defaults.
        if "metric" in data and not bool(data["metric"].item()):
            raise ValueError("Nonmetric geometry requires explicit geometry_metadata")
        return dict(
            pose_backend="legacy",
            metric=True,
            pose_units="metres",
            world_up_status="gravity_aligned",
            scale_status="legacy_metric",
            quality_status="legacy_unvalidated",
            world_inference_allowed=True,
        )
    meta = json.loads(str(data["geometry_metadata"].item()))
    if not isinstance(meta, dict) or meta.get("pose_frame") != "scene_camera":
        raise ValueError("Geometry metadata requires scene_camera poses")
    if type(meta.get("metric")) is not bool:
        raise ValueError("Geometry metric must be an explicit boolean")
    if "metric" in data and bool(data["metric"].item()) != meta["metric"]:
        raise ValueError("Conflicting metric flags in geometry")
    if meta.get("pose_backend") == "colmap":
        if meta.get("quality_status") in ("fail", "rejected"):
            raise ValueError("Rejected COLMAP geometry")
        allowed = _verified_colmap_world(meta)
        if meta.get("world_inference_allowed", False) and not allowed:
            raise ValueError(
                "COLMAP world inference requires verified scale, gravity and quality"
            )
        meta["world_inference_allowed"] = bool(allowed)
    return meta


def require_world_operations(meta, *, track=False, fuse=False, gaze_targets=False):
    if meta.get("pose_backend") == "colmap" and not _verified_colmap_world(meta):
        if track or fuse or gaze_targets:
            raise ValueError(
                "Unvalidated COLMAP scale/gravity: tracking, fusion and 3D gaze targets are disabled; use nonmetric preview"
            )
