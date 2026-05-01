package pas.risk.senses;

// SYSTEM IMPORTS
import edu.bu.jmat.Matrix;

import edu.bu.pas.risk.GameView;
import edu.bu.pas.risk.agent.senses.StateSensorArray;

// JAVA PROJECT IMPORTS

/**
 * A suite of sensors to convert a {@link GameView} into a feature vector (must
 * be a row-vector)
 */
public class MyStateSensorArray
        extends StateSensorArray {
    public static final int NUM_FEATURES = 1;

    public MyStateSensorArray(final int agentId) {
        super(agentId);
    }

    public Matrix getSensorValues(final GameView state) {
        return Matrix.randn(1, NUM_FEATURES); // row vector
    }

}
