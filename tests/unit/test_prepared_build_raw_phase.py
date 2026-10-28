"""Use already produced own-acquisition bridge records without remeasuring."""
from darsia.fluidflower.pipeline.build_raw_phase import BuildRawPhaseCollector
from .test_build_raw_phase import inputs


def test_complete_prepared_frames_skip_repeated_bridge_measurement(inputs):
    specs = [inputs.configuration["reference_spec"], *inputs.configuration["frames"]]
    prepared = {row["case_id"]: inputs.bridge.frame(row["raw_acquisition"],
        inputs.bridge.output / row["case_id"]) for row in specs}
    count = len([row for row in inputs.events if row[0] == "bridge"])
    collector = BuildRawPhaseCollector(inputs.bridge, inputs.configuration, inputs.output,
                                       prepared_frames=prepared)
    assert len([row for row in inputs.events if row[0] == "bridge"]) == count == 3
    assert collector.reference_bridge_frame is prepared[specs[0]["case_id"]]
    assert collector.reference_bridge_frame is collector.initial_previous_bridge_frame
    collector.close()
