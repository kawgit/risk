package pas.risk.senses;

import java.util.List;

// SYSTEM IMPORTS

// JAVA PROJECT IMPORTS
import edu.bu.jmat.Matrix;
import edu.bu.pas.risk.GameView;
import edu.bu.pas.risk.TerritoryOwnerView;
import edu.bu.pas.risk.agent.senses.StateSensorArray;
import edu.bu.pas.risk.territory.Continent;
import edu.bu.pas.risk.territory.Territory;
import edu.bu.pas.risk.territory.TerritoryCard;

/**
 * A suite of sensors to convert a {@link GameView} into a feature vector (must
 * be a row-vector)
 */
public class MyStateSensorArray
        extends StateSensorArray {
    // territory wise cnn features
    public static final int NUM_TERRITORIES = 42;
    public static final int NUM_CONTINENTS = 6;
    public static final int NUM_ALLEGIANCES = 2; // with us or against us (or neither)

    // feature counts: 1 log count + N bins via encodeCount
    public static final int NUM_ARMY_BINS = 6;
    public static final int NUM_MATCHING_CARD_BINS = 6;

    // general game information features -> now appended to each territory
    public static final int NUM_TOTAL_CARD_BINS = 6;
    public static final int NUM_WILD_CARD_BINS = 3;
    public static final int NUM_INFANTRY_CARD_BINS = 6;
    public static final int NUM_CAVALRY_CARD_BINS = 6;
    public static final int NUM_ARTILLERY_CARD_BINS = 6;
    public static final int NUM_ARMY_BUDGET_BINS = 6;
    public static final int NUM_ARMY_BONUS_BINS = 6;

    public static final int NUM_FEATURES_PER_TERRITORY = NUM_TERRITORIES + NUM_CONTINENTS + NUM_ALLEGIANCES
            + (1 + NUM_ARMY_BINS)
            + (1 + NUM_MATCHING_CARD_BINS)
            + (1 + NUM_TOTAL_CARD_BINS)
            + (1 + NUM_WILD_CARD_BINS)
            + (1 + NUM_INFANTRY_CARD_BINS)
            + (1 + NUM_CAVALRY_CARD_BINS)
            + (1 + NUM_ARTILLERY_CARD_BINS)
            + (1 + NUM_ARMY_BUDGET_BINS)
            + (1 + NUM_ARMY_BONUS_BINS);

    public static final int NUM_FEATURES = NUM_TERRITORIES * NUM_FEATURES_PER_TERRITORY + 1;

    public MyStateSensorArray(final int agentId) {
        super(agentId);
    }

    public Matrix getSensorValues(final GameView state) {
        Matrix result = Matrix.zeros(1, NUM_FEATURES);
        List<TerritoryCard> cards = state.getAgentInventory(this.getAgentId());

        for (int i = 0; i < NUM_TERRITORIES; i++) {
            int territoryId = i;

            Territory territory = state.getBoard().territories().getById(territoryId);
            Continent continent = territory.continent();
            TerritoryOwnerView ownerView = state.getTerritoryOwners().getById(territoryId);

            int continentId = continent.id();
            int allegianceId = ownerView.getOwner() == this.getAgentId() ? 1 : 0;
            int armyCount = ownerView.isUnclaimed() ? 0 : ownerView.getArmies();
            int matchesCardCount = (int) cards.stream()
                    .filter(card -> (!card.isWild() && card.territory().id() == territoryId)).count();

            int offset = territoryId * NUM_FEATURES_PER_TERRITORY;
            result.set(0, offset + territoryId, 1);
            offset += NUM_TERRITORIES;
            result.set(0, offset + continentId, 1);
            offset += NUM_CONTINENTS;
            if (!ownerView.isUnclaimed()) {
                result.set(0, offset + allegianceId, 1);
            }
            offset += NUM_ALLEGIANCES;

            offset = encodeCount(result, offset, NUM_ARMY_BINS, armyCount, true);
            offset = encodeCount(result, offset, NUM_MATCHING_CARD_BINS, matchesCardCount, true);
        }

        int totalCardCount = cards.size();
        int wildCardCount = (int) cards.stream().filter(card -> card.isWild()).count();
        int infantryCardCount = (int) cards.stream().filter(card -> (card.armyValue() == 1)).count();
        int cavalryCardCount = (int) cards.stream().filter(card -> (card.armyValue() == 2)).count();
        int artilleryCardCount = (int) cards.stream().filter(card -> (card.armyValue() == 3)).count();
        int armyBudgetCount = state.getArmyBudgets()[this.getAgentId()];
        int armyBonusCount = state.getBonusArmiesFor(this.getAgentId());

        int baseOffset = NUM_TERRITORIES + NUM_CONTINENTS + NUM_ALLEGIANCES
                + (1 + NUM_ARMY_BINS) + (1 + NUM_MATCHING_CARD_BINS);

        int offset = baseOffset;
        offset = encodeCount(result, offset, NUM_TOTAL_CARD_BINS, totalCardCount, true);
        offset = encodeCount(result, offset, NUM_WILD_CARD_BINS, wildCardCount, true);
        offset = encodeCount(result, offset, NUM_INFANTRY_CARD_BINS, infantryCardCount, true);
        offset = encodeCount(result, offset, NUM_CAVALRY_CARD_BINS, cavalryCardCount, true);
        offset = encodeCount(result, offset, NUM_ARTILLERY_CARD_BINS, artilleryCardCount, true);
        offset = encodeCount(result, offset, NUM_ARMY_BUDGET_BINS, armyBudgetCount, true);
        offset = encodeCount(result, offset, NUM_ARMY_BONUS_BINS, armyBonusCount, true);

        assert offset == NUM_FEATURES_PER_TERRITORY;

        for (int i = baseOffset; i < offset; i++) {
            broadcast(result, i, result.get(0, i));
        }

        result.set(0, NUM_FEATURES - 1, getStateReward(state));

        return result;
    }

    private static int encodeCount(Matrix result, int offset, int num_bins, int count, boolean log_scale) {
        result.set(0, offset, log_scale ? Math.log(1 + count) : count);
        offset++;
        int bin_idx = (int) Math.max(0, Math.min(num_bins - 1, count));
        result.set(0, offset + bin_idx, 1);
        offset += num_bins;
        return offset;
    }

    private static void broadcast(Matrix result, int offset, double value) {
        int num_features_per_territory = result.getShape().numCols() / 42;
        for (int i = offset; i < result.getShape().numCols(); i += num_features_per_territory) {
            result.set(0, i, value);
        }
    }

    private double getStateReward(final GameView state) {
        double armyDifference = 0;
        for (TerritoryOwnerView view : state.getTerritoryOwners()) {
            armyDifference += Math.pow(view.getArmies(), 1.2) * (view.getOwner() == this.getAgentId() ? 1 : -1);
        }
        return Math.max(-1000, Math.min(1000, armyDifference));
    }

}
