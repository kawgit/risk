package pas.risk.senses;

// SYSTEM IMPORTS

// JAVA PROJECT IMPORTS
import edu.bu.jmat.Matrix;
import edu.bu.pas.risk.GameView;
import edu.bu.pas.risk.agent.senses.PlacementSensorArray;
import edu.bu.pas.risk.territory.Territory;

/**
 * A suite of sensors to convert a {@link Territory} into a feature vector (must
 * be a row-vector)
 */
public class MyPlacementSensorArray
        extends PlacementSensorArray {

    public static final int NUM_FEATURES_PER_TERRITORY = 2; // is_target_territory, is_target_continent
    public static final int NUM_TERRITORIES = 42;
    public static final int NUM_FEATURES = NUM_TERRITORIES * NUM_FEATURES_PER_TERRITORY + 1;

    public MyPlacementSensorArray(final int agentId) {
        super(agentId);
    }

    public Matrix getSensorValues(final GameView state,
            final int numRemainingArmies,
            final Territory territory) {
        Matrix result = Matrix.zeros(1, NUM_FEATURES);

        int targetContinentId = territory.continent().id();
        int targetTerritoryId = territory.id();

        for (int i = 0; i < NUM_TERRITORIES; i++) {
            Territory t = state.getBoard().territories().getById(i);
            int offset = i * NUM_FEATURES_PER_TERRITORY;

            if (t.id() == targetTerritoryId) {
                result.set(0, offset, 1);
            }
            if (t.continent().id() == targetContinentId) {
                result.set(0, offset + 1, 1);
            }
        }

        result.set(0, NUM_FEATURES - 1, getBias(state, territory, this.getAgentId()));

        return result;
    }

    public static double getBias(GameView state, Territory territory, int agentId) {
        double bias = 1;
        int oldArmyCount = state.getTerritoryOwners().getById(territory.id()).getArmies();
        int newArmyCount = oldArmyCount + 1;

        // bonus for increasing concentration of forces
        double oldScore = Math.pow(oldArmyCount, 1.2);
        double newScore = Math.pow(newArmyCount, 1.2);

        bias += newScore - oldScore;

        return bias;
    }

}
