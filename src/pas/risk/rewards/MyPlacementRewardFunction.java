package pas.risk.rewards;

// SYSTEM IMPORTS

// JAVA PROJECT IMPORTS
import edu.bu.pas.risk.GameView;
import edu.bu.pas.risk.agent.rewards.RewardFunction;
import edu.bu.pas.risk.agent.rewards.RewardType;
import edu.bu.pas.risk.territory.Territory;

/**
 * <p>Represents a function which punishes/pleasures your model according to how well the {@link Territory}s its been
 * choosing to place armies have been. Your reward function could calculate R(s), R(s,t), or (R,t,a'): whichever
 * is easiest for you to think about (for instance does it make more sense to you to evaluate behavior when you see a
 * state, the action you took in that state, and how that action resolved? If so you want to pick R(s,t,s')).
 *
 * <p>By default this is configured to calculate R(s). If you want to change this you need to change the
 * {@link RewardType} enum in the constructor *and* you need to implement the corresponding method. Refer to
 * {@link RewardFunction} and {@link RewardType} for more details.
 */
public class MyPlacementRewardFunction
    extends RewardFunction<Territory>
{

    public MyPlacementRewardFunction(final int agentId)
    {
        super(RewardType.STATE, agentId); // change this enum if you don't want to do R(s)
    }

    public double getLowerBound() { return 0.0; }
    public double getUpperBound() { return 100.0; }

    /** {@inheritDoc} */
    public double getStateReward(final GameView state) { return 0.0; } // this sucks you'll need to change this

    /** {@inheritDoc} */
    public double getHalfTransitionReward(final GameView state,
                                          final Territory action) { return Double.NEGATIVE_INFINITY; }

    /** {@inheritDoc} */
    public double getFullTransitionReward(final GameView state,
                                          final Territory action,
                                          final GameView nextState) { return Double.NEGATIVE_INFINITY; }

}

