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
    public static final int NUM_FEATURES = NUM_TERRITORIES * NUM_FEATURES_PER_TERRITORY;

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

        return result;
    }

    public static int encodeCount(Matrix result, int offset, int num_bins, int count, boolean log_scale) {
        result.set(0, offset, log_scale ? Math.log(1 + count) : count);
        offset++;
        int bin_idx = (int) Math.max(0, Math.min(num_bins - 1, count));
        result.set(0, offset + bin_idx, 1);
        offset += num_bins;
        return offset;
    }

    public static void broadcast(Matrix result, int offset, double value) {
        int num_features_per_territory = result.getShape().numCols() / 42;
        for (int i = offset; i < result.getShape().numCols(); i += num_features_per_territory) {
            result.set(0, i, value);
        }
    }

}
