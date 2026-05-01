package pas.risk.senses;

// SYSTEM IMPORTS
import edu.bu.jmat.Matrix;

import edu.bu.pas.risk.GameView;
import edu.bu.pas.risk.action.Action;
import edu.bu.pas.risk.agent.senses.ActionSensorArray;

// JAVA PROJECT IMPORTS

/**
 * A suite of sensors to convert a {@link Action} into a feature vector (must be
 * a row-vector)
 */
public class MyActionSensorArray
        extends ActionSensorArray {

    public static final int NUM_FEATURES = 1;

    public MyActionSensorArray(final int agentId) {
        super(agentId);
    }

    public Matrix getSensorValues(final GameView state,
            final int actionCounter,
            final Action action) {
        return Matrix.randn(1, NUM_FEATURES); // row vector
    }

}
