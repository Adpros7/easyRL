from easyrl import EasyRL


def test_constructor_with_optional_params() -> None:
    agent = EasyRL(
        state_dim=4,
        action_dim=3,
        lr=1e-3,
        gamma=0.95,
        use_target_network=True,
        entropy_coef=0.01,
        clip_grad_norm=1.0,
        device="cpu",
    )
    assert agent.state_dim == 4
    assert agent.action_dim == 3
    assert agent.target_value_net is not None


def test_step_accepts_transition_and_returns_info() -> None:
    agent = EasyRL(state_dim=4, action_dim=2, lr=1e-3, gamma=0.99)
    info = agent.step(
        state=[0.0, 0.1, 0.2, 0.3],
        action=1,
        reward=1.0,
        next_state=[0.1, 0.2, 0.3, 0.4],
        done=False,
    )
    assert "loss" in info
    assert "td_error" in info
    assert len(agent.transitions) == 1


def test_select_action_output_type_and_range() -> None:
    agent = EasyRL(state_dim=4, action_dim=2, lr=1e-3, gamma=0.99)
    action = agent.select_action([0.1, -0.2, 0.3, 0.4], deterministic=True)
    assert isinstance(action, int)
    assert 0 <= action < 2
