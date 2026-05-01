package pas.risk.rewards;

// SYSTEM IMPORTS
import edu.bu.jmat.Pair;

import edu.bu.pas.risk.GameView;
import edu.bu.pas.risk.action.Action;
import edu.bu.pas.risk.agent.rewards.RewardFunction;
import edu.bu.pas.risk.agent.rewards.RewardType;

// JAVA PROJECT IMPORTS

/**
 * <p>
 * Represents a function which punishes/pleasures your model according to how
 * well the {@link Action}s its been
 * choosing have been. Your reward function could calculate R(s), R(s,a), or
 * (R,s,a'): whichever is easiest for you to
 * think about (for instance does it make more sense to you to evaluate behavior
 * when you see a state, the action you
 * took in that state, and how that action resolved? If so you want to pick
 * R(s,a,s')).
 *
 * <p>
 * By default this is configured to calculate R(s). If you want to change this
 * you need to change the
 * {@link RewardType} enum in the constructor *and* you need to implement the
 * corresponding method. Refer to
 * {@link RewardFunction} and {@link RewardType} for more details.
 */
public class MyActionRewardFunction
        extends RewardFunction<Action> {

    public MyActionRewardFunction(final int agentId) {
        super(RewardType.STATE, agentId); // change this enum if you don't want to do R(s)
    }

    public double getLowerBound() {
        return -10000.0;
    }

    public double getUpperBound() {
        return 10000.0;
    }

    /** {@inheritDoc} */
    public double getStateReward(final GameView state) {
        return 0.0;
    } // this sucks you'll need to change this

    /** {@inheritDoc} */
    public double getHalfTransitionReward(final GameView state,
            final Action action) {
        return Double.NEGATIVE_INFINITY;
    }

    /** {@inheritDoc} */
    public double getFullTransitionReward(final GameView state,
            final Action action,
            final GameView nextState) {
        return Double.NEGATIVE_INFINITY;
    }

}
