package pas.risk.agent;

import java.util.List;
import edu.bu.pas.risk.GameView;
import edu.bu.pas.risk.action.Action;
import edu.bu.pas.risk.territory.Territory;
import pas.risk.senses.MyActionSensorArray;
import pas.risk.senses.MyPlacementSensorArray;
import edu.bu.pas.risk.model.DualDecoderModel;

public class HeuristicAgent extends RiskQAgent {

    public HeuristicAgent(int agentId) {
        super(agentId);
    }

    @Override
    public DualDecoderModel initModel() {
        return null; // No neural network component
    }

    private <T> T chooseSoftmax(GameView game, List<T> options, BiasFunction<T> biasFunc) {
        if (options == null || options.isEmpty())
            return null;
        if (options.size() == 1)
            return options.get(0);

        double[] logits = new double[options.size()];
        for (int i = 0; i < options.size(); i++) {
            logits[i] = biasFunc.apply(game, options.get(i), this.agentId());
        }

        return this.chooseRandomWithLogits(options, logits, 0.01);
    }

    @Override
    public Action getExplorationRedeemAction(GameView game, int actionCounter, boolean canRedeemCards) {
        List<Action> options = this.getRedeemActions(game, actionCounter, canRedeemCards,
                game.getAgentInventory(this.agentId()).size() < 5);
        return chooseSoftmax(game, options, MyActionSensorArray::getBias);
    }

    @Override
    public boolean shouldExploreRedeemMovePhase(GameView game, int actionCounter, boolean canRedeemCards) {
        return true;
    }

    @Override
    public Action getExplorationAttackActionRedeemIfForced(GameView game, int actionCounter, boolean canRedeemCards) {
        List<Action> options = this.getAttackRedeemActions(game, actionCounter, canRedeemCards);
        return chooseSoftmax(game, options, MyActionSensorArray::getBias);
    }

    @Override
    public boolean shouldExploreAttackRedeemIfForcedMovePhase(GameView game, int actionCounter,
            boolean canRedeemCards) {
        return true;
    }

    @Override
    public Action getExplorationFortifySkipAction(GameView game, int actionCounter, boolean canRedeemCards) {
        List<Action> options = this.getFortifyActions(game, actionCounter, canRedeemCards);
        return chooseSoftmax(game, options, MyActionSensorArray::getBias);
    }

    @Override
    public boolean shouldExploreFortifySkipMovePhase(GameView game, int actionCounter, boolean canRedeemCards) {
        return true;
    }

    @Override
    public Territory getExplorationPlacement(GameView game, boolean isDuringSetup, int remainingArmies) {
        List<Territory> options = this.getPotentialPlacements(game, isDuringSetup, remainingArmies);
        return chooseSoftmax(game, options, MyPlacementSensorArray::getBias);
    }

    @Override
    public boolean shouldExplorePlacementPhase(GameView game, boolean isDuringSetup, int remainingArmies) {
        return true;
    }
}
