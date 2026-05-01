package pas.risk.senses;

// SYSTEM IMPORTS
import edu.bu.jmat.Matrix;

import edu.bu.pas.risk.GameView;
import edu.bu.pas.risk.agent.senses.PlacementSensorArray;
import edu.bu.pas.risk.territory.Territory;

// JAVA PROJECT IMPORTS

/**
 * A suite of sensors to convert a {@link Territory} into a feature vector (must
 * be a row-vector)
 */
public class MyPlacementSensorArray
        extends PlacementSensorArray {

    public static final int NUM_FEATURES = 1;

    public MyPlacementSensorArray(final int agentId) {
        super(agentId);
    }

    public Matrix getSensorValues(final GameView state,
            final int numRemainingArmies,
            final Territory territory) {
        return Matrix.randn(1, NUM_FEATURES); // row vector
    }

}
