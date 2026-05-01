package pas.risk.rewards;

// SYSTEM IMPORTS

// JAVA PROJECT IMPORTS
import edu.bu.pas.risk.GameView;
import edu.bu.pas.risk.TerritoryOwnerView;
import edu.bu.pas.risk.action.Action;
import edu.bu.pas.risk.agent.rewards.RewardFunction;
import edu.bu.pas.risk.agent.rewards.RewardType;

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

    public static final double RANGE = 100;
    public static final double FORCE_END_COEFFICIENT = .1;

    public MyActionRewardFunction(final int agentId) {
        super(RewardType.STATE, agentId); // change this enum if you don't want to do R(s)
    }

    public double getLowerBound() {
        return 2 * -RANGE;
    }

    public double getUpperBound() {
        return 2 * RANGE;
    }

    /** {@inheritDoc} */
    public double getStateReward(final GameView state) {
        double[] armiesPerOwner = new double[state.getNumAgents()];
        for (TerritoryOwnerView view : state.getTerritoryOwners()) {
            if (view.isUnclaimed()) {
                continue;
            }
            armiesPerOwner[view.getOwner()] += Math.pow(view.getArmies(), 1.2);
        }

        double ourArmies = armiesPerOwner[this.getAgentId()];
        double theirArmies = 0;
        for (int i = 0; i < state.getNumAgents(); i++) {
            if (i == this.getAgentId())
                continue;
            theirArmies = Math.max(theirArmies, armiesPerOwner[i]);
        }

        double reward = Math.max(-RANGE - FORCE_END_COEFFICIENT * theirArmies,
                Math.min(RANGE - FORCE_END_COEFFICIENT * theirArmies, ourArmies - theirArmies));
        return Math.max(getLowerBound(), Math.min(getUpperBound(), reward));
    }

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
