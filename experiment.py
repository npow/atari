import rlglue.RLGlue as RLGlue
import argparse
import sys

NUM_EPOCHS = 100
EPOCH_LENGTH = 50000
TEST_LENGTH = 10000

def run_epoch(epoch, num_steps, prefix):
    steps_left = num_steps
    while steps_left > 0:
        print prefix + " epoch: ", epoch, "steps_left: ", steps_left
        terminal = RLGlue.RL_episode(steps_left)
        if not terminal:
            RLGlue.RL_agent_message("episode_end")
        steps_left -= RLGlue.RL_num_steps()

def main():
    RLGlue.RL_init()
    is_testing = len(sys.argv) > 1 and sys.argv[1] == 'test'
    for epoch in xrange(NUM_EPOCHS):
        if is_testing:
            RLGlue.RL_agent_message("start_testing")
            run_epoch(epoch, TEST_LENGTH, "testing")
            RLGlue.RL_agent_message("finish_testing " + str(epoch))
        else:
            run_epoch(epoch, EPOCH_LENGTH, "training")
            RLGlue.RL_agent_message("finish_epoch " + str(epoch))

if __name__ == "__main__":
    main()
