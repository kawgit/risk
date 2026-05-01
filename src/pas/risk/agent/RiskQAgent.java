package pas.risk.agent;

// SYSTEM IMPORTS
import edu.bu.jnn.layers.*;
import edu.bu.jnn.models.Sequential;

import edu.bu.pas.risk.GameView;
import edu.bu.pas.risk.action.Action;
import edu.bu.pas.risk.agent.NeuralQAgent;
import edu.bu.pas.risk.agent.rewards.RewardFunction;
import edu.bu.pas.risk.agent.senses.*;
import edu.bu.pas.risk.model.DualDecoderModel;
import edu.bu.pas.risk.territory.Territory;

import java.util.Collections;
import java.util.List;
import java.util.Random;

// JAVA PROJECT IMPORTS
import pas.risk.rewards.MyActionRewardFunction;
import pas.risk.rewards.MyPlacementRewardFunction;
import pas.risk.senses.MyActionSensorArray;
import pas.risk.senses.MyPlacementSensorArray;
import pas.risk.senses.MyStateSensorArray;

/**
 * Represents a {@link NeuralQAgent} where all of the configuration options are
 * specified. These configuration options
 * are:
 * <ol>
 * <li>The architecture of the {@link DualDecoderModel} we're using for this
 * assignment. More specifically, what
 * is the architecture of the encoder, the action decoder, and the placement
 * decoder?</li>
 * <li>How is a state (e.g. a {@link GameView}) perceived by the model? This is
 * done via a
 * {@link MyStateSensorArray} object which is responsible for converting a
 * {@link GameView} into a feature
 * vector which *must* be a row-vector.</li>
 * <li>How is an {@link Action} perceived by the model? This is done via a
 * {@link MyActionSensorArray} object which is responsible for converting a
 * {@link Action} into a feature
 * vector which *must* be a row-vector.</li>
 * <li>How is a {@link Territory} perceived by the model? This is done via a
 * {@link MyPlacementSensorArray} object which is responsible for converting a
 * {@link Territory} into a feature
 * vector which *must* be a row-vector.</li>
 * <li>How is the model punished/pleasured according to the quality of
 * {@link Action}s that it chooses? This
 * is done via a {@link MyActionRewardFunction} which you can configure to
 * calculate R(s), R(s,a),
 * or R(s,a,s')</li>
 * <li>How is the model punished/pleasured according to the quality of
 * {@link Territory}s that it chooses to place
 * armies at? This is done via a {@link MyPlacementRewardFunction} which you can
 * configure
 * to calculate R(s), R(s,t), or R(s,t,s')</li>
 * </ol>
 *
 */
public class RiskQAgent
        extends NeuralQAgent {

    public RiskQAgent(int agentId) {
        super(agentId);
    }

    /**
     * A method to create your neural network architecture. This is done by making
     * three separate {@link Sequential}
     * instances (with appropriate dimensions) and then chucking them into the
     * {@link DualDecoderModel} class I made
     * for you which coordinates them.
     *
     * @return The {@link DualDecoderModel} which coordinates the three neural
     *         networks you make here.
     */
    public DualDecoderModel initModel() {
        Sequential encoder = new Sequential();
        Sequential actionDecoder = new Sequential();
        Sequential placementDecoder = new Sequential();

        return new DualDecoderModel(encoder, actionDecoder, placementDecoder);
    }

    /**
     * A method to create your state sensor suite.
     *
     * @return Your state sensor suite
     */
    @Override
    public StateSensorArray createStateSensors() {
        return new MyStateSensorArray(this.agentId());
    }

    /**
     * A method to create your action sensor suite.
     *
     * @return Your action sensor suite
     */
    @Override
    public ActionSensorArray createActionSensors() {
        return new MyActionSensorArray(this.agentId());
    }

    /**
     * A method to create your placement sensor suite.
     *
     * @return Your placement sensor suite
     */
    @Override
    public PlacementSensorArray createPlacementSensors() {
        return new MyPlacementSensorArray(this.agentId());
    }

    /**
     * A method to create your action reward function.
     *
     * @return Your action reward function
     */
    @Override
    public RewardFunction<Action> createActionReward() {
        return new MyActionRewardFunction(this.agentId());
    }

    /**
     * A method to create your placement reward function.
     *
     * @return Your placement reward function
     */
    @Override
    public RewardFunction<Territory> createPlacementReward() {
        return new MyPlacementRewardFunction(this.agentId());
    }

    public static <T> T chooseRandom(final List<T> list,
            final Random random) {
        return list.get(random.nextInt(list.size()));
    }

    /**
     * A method to choose an {@link Action} when it is in the redeem phase of a
     * turn. You are free to write your own
     * code to choose which move to explore however your decision should be
     * stochastic (e.g. determinism is bad).
     *
     * @param game           the current state of the game
     * @param actionCounter  how many actions you've made so far in this turn
     * @param canRedeemCards can you redeem cards
     * @return the {@link Action} to do
     */
    @Override
    public Action getExplorationRedeemAction(final GameView game,
            final int actionCounter,
            final boolean canRedeemCards) {
        final List<Action> options = this.getRedeemActions(game, actionCounter, canRedeemCards,
                game.getAgentInventory(this.agentId()).size() < 5);
        return chooseRandom(options, new Random());
    }

    /**
     * A method to decide whether to listen to your q-function or not. This will be
     * called ever time your agent
     * needs to decide what move to make in the redeem phase of your turn.
     *
     * @param game           the current state of the game
     * @param actionCounter  how many actions you've made so far in this turn
     * @param canRedeemCards can you redeem cards
     * @return <code>true</code> if <code>getExplorationRedeemAction</code> should
     *         be called or if your action
     *         q-function should be argmaxed.
     */
    @Override
    public boolean shouldExploreRedeemMovePhase(final GameView game,
            final int actionCounter,
            final boolean canRedeemCards) {
        return false;
    }

    /**
     * A method to choose an {@link Action} when it is in the attacking phase of a
     * turn. You are free to write your own
     * code to choose which move to explore however your decision should be
     * stochastic (e.g. determinism is bad).
     *
     * @param game           the current state of the game
     * @param actionCounter  how many actions you've made so far in this turn
     * @param canRedeemCards can you redeem cards
     * @return the {@link Action} to do
     */
    @Override
    public Action getExplorationAttackActionRedeemIfForced(final GameView game,
            final int actionCounter,
            final boolean canRedeemCards) {
        final List<Action> options = this.getAttackRedeemActions(game, actionCounter, canRedeemCards);
        return chooseRandom(options, new Random());
    }

    /**
     * A method to decide whether to listen to your q-function or not. This will be
     * called ever time your agent
     * needs to decide what move to make in the attacking phase of your turn.
     *
     * @param game           the current state of the game
     * @param actionCounter  how many actions you've made so far in this turn
     * @param canRedeemCards can you redeem cards
     * @return <code>true</code> if
     *         <code>getExplorationAttackActionRedeemIfForced</code> should be
     *         called or
     *         if your action q-function should be argmaxed.
     */
    @Override
    public boolean shouldExploreAttackRedeemIfForcedMovePhase(final GameView game,
            final int actionCounter,
            final boolean canRedeemCards) {
        return false;
    }

    /**
     * A method to choose an {@link Action} when it is in the fortifying phase of a
     * turn. You are free to write your own
     * code to choose which move to explore however your decision should be
     * stochastic (e.g. determinism is bad).
     *
     * @param game           the current state of the game
     * @param actionCounter  how many actions you've made so far in this turn
     * @param canRedeemCards can you redeem cards
     * @return the {@link Action} to do
     */
    @Override
    public Action getExplorationFortifySkipAction(final GameView game,
            final int actionCounter,
            final boolean canRedeemCards) {
        final List<Action> options = this.getFortifyActions(game, actionCounter, canRedeemCards);
        return chooseRandom(options, new Random());
    }

    /**
     * A method to decide whether to listen to your q-function or not. This will be
     * called ever time your agent
     * needs to decide what move to make in the fortifying phase of your turn.
     *
     * @param game           the current state of the game
     * @param actionCounter  how many actions you've made so far in this turn
     * @param canRedeemCards can you redeem cards
     * @return <code>true</code> if <code>getExplorationFortifySkipAction</code>
     *         should be called or
     *         if your action q-function should be argmaxed.
     */
    @Override
    public boolean shouldExploreFortifySkipMovePhase(final GameView game,
            final int actionCounter,
            final boolean canRedeemCards) {
        return false;
    }

    /**
     * A method to choose an {@link Territory} when it is in the army placing phase
     * of a turn (or during game setup).
     * You are free to write your own code to choose which move to explore however
     * your decision should be stochastic
     * (e.g. determinism is bad).
     *
     * @param game            the current state of the game
     * @param isDuringSetup   is this during the game setup or at the beginning of
     *                        your move
     * @param remainingArmies number of armies left to place
     * @return the {@link Territory} to place an army at
     */
    @Override
    public Territory getExplorationPlacement(final GameView game,
            final boolean isDuringSetup,
            final int remainingArmies) {
        final List<Territory> options = this.getPotentialPlacements(game, isDuringSetup, remainingArmies);
        return chooseRandom(options, new Random());
    }

    /**
     * A method to decide whether to listen to your q-function or not. This will be
     * called ever time your agent
     * needs to decide what {@link Territory} to place an army at.
     *
     * @param game            the current state of the game
     * @param isDuringSetup   is this during the game setup or at the beginning of
     *                        your move
     * @param remainingArmies number of armies left to place
     * @return <code>true</code> if <code>getExplorationPlacement</code> should be
     *         called or
     *         if your action q-function should be argmaxed.
     */
    @Override
    public boolean shouldExplorePlacementPhase(final GameView game,
            final boolean isDuringSetup,
            final int remainingArmies) {
        return false;
    }

    private static int lastPrintedTurn = -1;
    private int numGames = 0;
    private int numWins = 0;
    private int numTurns = 0;

    @Override
    public void onTurnEnd(GameView game, int agentIdx) {
        super.onTurnEnd(game, agentIdx);

        int currentTurn = game.getNumTurns();
        if (currentTurn != lastPrintedTurn) {
            lastPrintedTurn = currentTurn;

            if (game.isOver()) {
                numGames++;
                if (!game.getTerritoriesOwnedBy(this.agentId()).isEmpty()) {
                    System.out.println("WIN");
                    numWins++;
                } else {
                    System.out.println("LOSS");
                }
                numTurns += game.getNumTurns();

                System.out.println("Over " + numGames + " games:");
                System.out.println("  Win Rate: " + (double) numWins / numGames);
                System.out.println("  Avg Length: " + (double) numTurns / numGames);
                System.out.println();
            }
        }

    }
}
