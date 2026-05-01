package pas.risk.senses;

// SYSTEM IMPORTS
import edu.bu.jmat.Matrix;

import edu.bu.pas.risk.GameView;
import edu.bu.pas.risk.TerritoryOwnerView;
import edu.bu.pas.risk.action.Action;
import edu.bu.pas.risk.action.AttackAction;
import edu.bu.pas.risk.action.FortifyAction;
import edu.bu.pas.risk.action.NoAction;
import edu.bu.pas.risk.action.RedeemCardsAction;
import edu.bu.pas.risk.agent.senses.ActionSensorArray;
import edu.bu.pas.risk.territory.Territory;

// JAVA PROJECT IMPORTS

/**
 * A suite of sensors to convert a {@link Action} into a feature vector (must be
 * a row-vector)
 */
public class MyActionSensorArray
        extends ActionSensorArray {

    public static final int NUM_FEATURES = 1;
    private GameView stateAtStartOfTurn = null;

    public MyActionSensorArray(final int agentId) {
        super(agentId);
    }

    public Matrix getSensorValues(final GameView state,
            final int actionCounter,
            final Action action) {

        if (stateAtStartOfTurn == null || state.getNumTurns() != stateAtStartOfTurn.getNumTurns()) {
            stateAtStartOfTurn = state;
        }

        double result = 0;

        boolean hasTakenTerritoryThisTurn = stateAtStartOfTurn.getTerritoriesOwnedBy(this.getAgentId()).size() < state
                .getTerritoriesOwnedBy(this.getAgentId()).size();

        if (!hasTakenTerritoryThisTurn && !(action instanceof AttackAction)) {
            result -= 1000;
        }

        String phase = getPhase(state);

        if (action instanceof AttackAction) {
            AttackAction attack = (AttackAction) action;
            
            if (phase.equals("EARLY")) {
                result -= 100;

                if (attack.attackingArmies() != 3) {
                    result -= 1000;
                }

                result -= state.getTerritoryOwners().get(attack.to()).getArmies();
                result += attack.movingArmies();

                result += attack.to().continent().id() == 5 ? 10 : 0;
            }
        } else if (action instanceof RedeemCardsAction) {
            if (phase.equals("EARLY")) {
                result -= 1000;
            }
        } else if (action instanceof FortifyAction) {

        }

        Matrix resultMatrix = Matrix.zeros(1, 1);
        resultMatrix.set(0, 0, result + Math.random() * .000000001);
        return resultMatrix;
    }

    private String getPhase(final GameView state) {
        /*
         * EARLY: 0-5 redeemptions
         * LATE: 6+ redemptions
         */
        return state.getNumPreviousRedemptions() >= 6 ? "LATE" : "EARLY";
    }
}
