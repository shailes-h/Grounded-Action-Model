"""End-to-end agent test with weights (skipped when GAM_TEST_POLICY / GAM_TEST_WILDDET are unset)."""
import os

import numpy as np
import pytest

POLICY = os.environ.get("GAM_TEST_POLICY")
WILDDET = os.environ.get("GAM_TEST_WILDDET")
SAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "sample_frame.npz")


@pytest.mark.skipif(not (POLICY and WILDDET), reason="set GAM_TEST_POLICY and GAM_TEST_WILDDET")
def test_agent_two_steps():
    from gam.deploy.agent import AgentConfig, GAMAgent
    from gam.deploy.interfaces import Frame
    s = np.load(SAMPLE, allow_pickle=True)
    click = s["ref_click"]
    agent = GAMAgent(AgentConfig(policy_ckpt=POLICY, wilddet_ckpt=WILDDET,
                                 planner_url=f"mock:{click[0]},{click[1]}", observe_every=0, seed=0))
    agent.reset(str(s["instruction"]))
    frame = Frame(rgb=s["rgb"], depth_m=s["depth_m"], K=s["K"], T_cam2base=s["T_cam2base"])
    agent.push_state(s["q_prev"]); agent.push_state(s["q"])
    a = agent.act(frame, s["q"], arm_mask=s["arm_mask"].astype(bool))
    assert a.shape == (15, 14) and np.isfinite(a).all()
    assert "slot0" in agent.last["objects"] and agent.last["patch_mask"].sum() > 0
    # matches the recorded demonstration to within a few hundredths of a radian on this frame
    assert np.abs(a - s["ref_action"][:15]).mean() < 0.05
    a2 = agent.act(frame, s["q"], arm_mask=s["arm_mask"].astype(bool))
    assert a2.shape == (15, 14)
