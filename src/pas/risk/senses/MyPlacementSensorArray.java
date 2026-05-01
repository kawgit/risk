package pas.risk.senses;

// SYSTEM IMPORTS
import edu.bu.jmat.Matrix;

import edu.bu.pas.risk.GameView;
import edu.bu.pas.risk.TerritoryOwnerView;
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

        final String phase = getPhase(state);
        final TerritoryOwnerView territoryOwnersView = state.getTerritoryOwners().get(territory);

        double result = 0;
        if (phase.equals("CLAIMING")) {

            // continent bonus
            final double[] continentValues = new double[] {
                    1, // Asia
                    3, // North America
                    4, // South America
                    2, // Africa
                    0, // Europe
                    5 // Australia
            };
            int continentId = territory.continent().id();
            result += continentValues[continentId];

            // unclaimed/enemy neighbor penalty
            for (Territory neighbor : territory.adjacentTerritories()) {
                TerritoryOwnerView neighborOwnerView = state.getTerritoryOwners().get(neighbor);
                if (neighborOwnerView.isUnclaimed()) {
                    result -= 0.1;
                } else if (neighborOwnerView.getOwner() != this.getAgentId()) {
                    result -= 1;
                }
            }
        } else {
            // penalize unthreatened territories
            boolean isThreatened = false;
            for (Territory neighbor : territory.adjacentTerritories()) {
                TerritoryOwnerView neighborOwnerView = state.getTerritoryOwners().get(neighbor);
                if (neighborOwnerView.getOwner() != this.getAgentId()) {
                    isThreatened = true;
                    break;
                }
            }
            if (!isThreatened) {
                result -= 10000;
            }
        }

        if (phase.equals("PLACING")) {
            // spread out armies
            result -= territoryOwnersView.getArmies() / 10.0;

        } else if (phase.equals("PLAYING")) {

        }

        Matrix resultMatrix = Matrix.zeros(1, 1);
        resultMatrix.set(0, 0, result + Math.random() * .000000001);
        return resultMatrix;
    }

    private String getPhase(final GameView state) {

        /*
         * CLAIMING: we are claiming territories at the start of the game
         * PLACING: we are placing the remaining armies at the start of the game
         * PLAYING: we are playing the game and are in the reinforcement phase of turn
         */

        final int[] numStartingArmiesLookup = { -1, -1, 40, 35, 30, 25, 20 };

        final int numStartingArmies = numStartingArmiesLookup[state.getNumAgents()];

        for (Territory territory : state.getBoard().territories()) {
            if (state.getTerritoryOwners().get(territory).isUnclaimed()) {
                return "CLAIMING";
            }
        }

        if (state.getNumTurns() > 0) {
            return "PLAYING";
        }

        int numArmies = 0;
        for (Territory territory : state.getTerritoriesOwnedBy(this.getAgentId())) {
            numArmies += state.getTerritoryOwners().get(territory).getArmies();
        }

        if (numArmies < numStartingArmies) {
            return "PLACING";
        }

        // edge case where it's turn 0 and we are going first
        return "PLAYING";
    }

}
