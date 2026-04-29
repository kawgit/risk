package pas.risk.senses;

import java.util.Set;

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
import edu.bu.pas.risk.territory.TerritoryCard;

/**
 * A suite of sensors to convert a {@link Action} into a feature vector (must be
 * a row-vector)
 */
public class MyActionSensorArray extends ActionSensorArray {

    public static final int MOVING_ARMIES_BINS = 6;
    public static final int ATTACK_ARMIES_BINS = 4;
    public static final int REDEMPTION_BINS = 6;

    // Attack: flag (1), src (1), dst (1), moving (1+6=7), attacking (1+4=5)
    public static final int NUM_ATTACK_FEATURES = 1 + 1 + 1 + (1 + MOVING_ARMIES_BINS) + (1 + ATTACK_ARMIES_BINS);
    // Fortify: flag (1), src (1), dst (1), moving (1+6=7)
    public static final int NUM_FORTIFY_FEATURES = 1 + 1 + 1 + (1 + MOVING_ARMIES_BINS);
    // Redeem: flag (1), match (1), redemption (1+6=7)
    public static final int NUM_REDEEM_FEATURES = 1 + 1 + (1 + REDEMPTION_BINS);
    // NoAction: flag (1)
    public static final int NUM_NO_ACTION_FEATURES = 1;

    public static final int NUM_FEATURES_PER_TERRITORY = NUM_ATTACK_FEATURES + NUM_FORTIFY_FEATURES
            + NUM_REDEEM_FEATURES + NUM_NO_ACTION_FEATURES;
    public static final int NUM_TERRITORIES = 42;
    public static final int NUM_FEATURES = NUM_TERRITORIES * NUM_FEATURES_PER_TERRITORY + 1;

    public MyActionSensorArray(final int agentId) {
        super(agentId);
    }

    public Matrix getSensorValues(final GameView state,
            final int actionCounter,
            final Action action) {
        Matrix result = Matrix.zeros(1, NUM_FEATURES);

        boolean isAttack = action instanceof AttackAction;
        boolean isFortify = action instanceof FortifyAction;
        boolean isRedeem = action instanceof RedeemCardsAction;
        boolean isNoAction = action instanceof NoAction;

        AttackAction attack = isAttack ? (AttackAction) action : null;
        FortifyAction fortify = isFortify ? (FortifyAction) action : null;
        RedeemCardsAction redeem = isRedeem ? (RedeemCardsAction) action : null;

        int tradeInNum = 1 + state.getNumPreviousRedemptions();
        int redemptionAmount = isRedeem ? TerritoryCard.getRedemptionAmount(tradeInNum) : 0;

        for (int i = 0; i < NUM_TERRITORIES; i++) {
            int territoryId = i;
            int offset = territoryId * NUM_FEATURES_PER_TERRITORY;

            // --- ATTACK SECTION ---
            if (isAttack) {
                result.set(0, offset, 1);
                result.set(0, offset + 1, attack.from().id() == territoryId ? 1 : 0);
                result.set(0, offset + 2, attack.to().id() == territoryId ? 1 : 0);
                int localOffset = offset + 3;
                localOffset = encodeCount(result, localOffset, MOVING_ARMIES_BINS, attack.movingArmies(), true);
                localOffset = encodeCount(result, localOffset, ATTACK_ARMIES_BINS, attack.attackingArmies(),
                        true);
                assert localOffset == offset + NUM_ATTACK_FEATURES;
            }
            offset += NUM_ATTACK_FEATURES;

            // --- FORTIFY SECTION ---
            if (isFortify) {
                result.set(0, offset, 1);
                result.set(0, offset + 1, fortify.from().id() == territoryId ? 1 : 0);
                result.set(0, offset + 2, fortify.to().id() == territoryId ? 1 : 0);
                int localOffset = offset + 3;
                localOffset = encodeCount(result, localOffset, MOVING_ARMIES_BINS, fortify.deltaArmies(), true);
                assert localOffset == offset + NUM_FORTIFY_FEATURES;
            }
            offset += NUM_FORTIFY_FEATURES;

            // --- REDEEM SECTION ---
            if (isRedeem) {
                result.set(0, offset, 1);

                int matches = 0;
                if (!redeem.card1().isWild() && redeem.card1().territory().id() == territoryId)
                    matches++;
                if (!redeem.card2().isWild() && redeem.card2().territory().id() == territoryId)
                    matches++;
                if (!redeem.card3().isWild() && redeem.card3().territory().id() == territoryId)
                    matches++;

                result.set(0, offset + 1, matches);

                int localOffset = offset + 2;
                localOffset = encodeCount(result, localOffset, REDEMPTION_BINS, redemptionAmount, true);
                assert localOffset == offset + NUM_REDEEM_FEATURES;
            }
            offset += NUM_REDEEM_FEATURES;

            // --- NO ACTION SECTION ---
            if (isNoAction) {
                result.set(0, offset, 1);
            }
            offset += NUM_NO_ACTION_FEATURES;

            assert offset == (territoryId + 1) * NUM_FEATURES_PER_TERRITORY;
        }

        result.set(0, NUM_FEATURES - 1, getBias(state, action, this.getAgentId()));

        return result;
    }

    public static double getBias(GameView state, Action action, int agentId) {
        // Bias calculation for actions should encode the expected deviation of future
        // state rewards from current state reward

        double bias = 0;
        if (action instanceof AttackAction) {
            AttackAction attack = (AttackAction) action;
            // bonus for choosing greedily beneficial attacks
            final double[][] expectedNetChange = {
                    // 1 Def // 2 Def
                    { -0.167, -0.491 }, // 1 Attacker Die
                    { 0.157, -0.441 }, // 2 Attacker Dice
                    { 0.319, 0.158 } // 3 Attacker Dice
            };

            TerritoryOwnerView toView = state.getTerritoryOwners().getById(attack.to().id());

            int attackers = attack.attackingArmies();
            int defenders = Math.min(toView.getArmies(), 2);

            assert attackers >= 1 && attackers <= 3;
            assert defenders >= 1 && defenders <= 2;

            bias += expectedNetChange[attackers - 1][defenders - 1];

            // small bonus for keeping forces together

            int fromArmies = state.getTerritoryOwners().getById(attack.from().id()).getArmies();
            int movingArmies = attack.movingArmies();

            bias += (movingArmies == fromArmies - 1 || movingArmies == 1) ? 0.05 : 0;

        } else if (action instanceof FortifyAction) {
            FortifyAction fortify = (FortifyAction) action;
            // bonus for increasing concentration of forces
            TerritoryOwnerView fromView = state.getTerritoryOwners().getById(fortify.from().id());
            TerritoryOwnerView toView = state.getTerritoryOwners().getById(fortify.to().id());

            int oldFromArmies = fromView.getArmies();
            int oldToArmies = toView.getArmies();

            int newFromArmies = oldFromArmies - fortify.deltaArmies();
            int newToArmies = oldToArmies + fortify.deltaArmies();

            double oldScore = Math.pow(oldFromArmies, 1.2) + Math.pow(oldToArmies, 1.2);
            double newScore = Math.pow(newFromArmies, 1.2) + Math.pow(newToArmies, 1.2);

            bias += newScore - oldScore;

            // small bonus for reinforcing borders
            boolean anyAdjacentEnemy = false;
            for (Territory t : fortify.to().adjacentTerritories()) {
                TerritoryOwnerView tView = state.getTerritoryOwners().getById(t.id());
                if (tView.getOwner() != agentId) {
                    anyAdjacentEnemy = true;
                    break;
                }
            }
            bias += anyAdjacentEnemy ? 0.05 : 0;

        } else if (action instanceof RedeemCardsAction) {
            RedeemCardsAction redeem = (RedeemCardsAction) action;
            Set<Integer> cardIds = Set.of(
                    redeem.card1().isWild() ? -1 : redeem.card1().territory().id(),
                    redeem.card2().isWild() ? -1 : redeem.card2().territory().id(),
                    redeem.card3().isWild() ? -1 : redeem.card3().territory().id());
            // bonus for redemption amount
            int redemptionCount = 1 + state.getNumPreviousRedemptions();
            int redemptionAmount = TerritoryCard.getRedemptionAmount(redemptionCount)
                    + (int) (state.getTerritoriesOwnedBy(agentId).stream().filter(t -> cardIds.contains(t.id())).count()
                            * 2);
            bias += redemptionAmount;
        }

        return bias;
    }

    private static int encodeCount(Matrix result, int offset, int num_bins, int count, boolean log_scale) {
        result.set(0, offset, log_scale ? Math.log(1 + count) : count);
        offset++;
        int bin_idx = (int) Math.max(0, Math.min(num_bins - 1, count));
        result.set(0, offset + bin_idx, 1);
        offset += num_bins;
        return offset;
    }

}
