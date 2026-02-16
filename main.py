from easyrl import EasyRL


def main() -> None:
    agent = EasyRL(state_dim=4, action_dim=2, lr=1e-3, gamma=0.99, entropy_coef=0.01)
    state = [0.1, 0.2, -0.1, 0.0]
    action = agent.select_action(state)
    info = agent.step(state=state, action=action, reward=1.0, next_state=state, done=False)
    print(f"Action: {action}, step info: {info}")


if __name__ == "__main__":
    main()
