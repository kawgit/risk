package pas.risk.agent;

// SYSTEM IMPORTS
import java.util.*;

// JAVA PROJECT IMPORTS
import edu.bu.jmat.Matrix;
import edu.bu.jnn.models.Sequential;
import edu.bu.jnn.Module;
import edu.bu.jnn.Parameter;
import edu.bu.pas.risk.action.Action;
import edu.bu.pas.risk.agent.NeuralQAgent;
import edu.bu.pas.risk.agent.rewards.RewardFunction;
import edu.bu.pas.risk.agent.senses.*;
import edu.bu.pas.risk.GameView;
import edu.bu.pas.risk.model.DualDecoderModel;
import edu.bu.pas.risk.territory.Territory;
import pas.risk.rewards.MyActionRewardFunction;
import pas.risk.rewards.MyPlacementRewardFunction;
import pas.risk.senses.MyActionSensorArray;
import pas.risk.senses.MyPlacementSensorArray;
import pas.risk.senses.MyStateSensorArray;

/**
 * Represents a {@link NeuralQAgent} where all of the configuration options are
 * specified. These configuration options
 * are:
 * <ol>
 * <li>The architecture of the {@link DualDecoderModel} we're using for this
 * assignment. More specifically, what
 * is the architecture of the encoder, the action decoder, and the placement
 * decoder?</li>
 * <li>How is a state (e.g. a {@link GameView}) perceived by the model? This is
 * done via a
 * {@link MyStateSensorArray} object which is responsible for converting a
 * {@link GameView} into a feature
 * vector which *must* be a row-vector.</li>
 * <li>How is an {@link Action} perceived by the model? This is done via a
 * {@link MyActionSensorArray} object which is responsible for converting a
 * {@link Action} into a feature
 * vector which *must* be a row-vector.</li>
 * <li>How is a {@link Territory} perceived by the model? This is done via a
 * {@link MyPlacementSensorArray} object which is responsible for converting a
 * {@link Territory} into a feature
 * vector which *must* be a row-vector.</li>
 * <li>How is the model punished/pleasured according to the quality of
 * {@link Action}s that it chooses? This
 * is done via a {@link MyActionRewardFunction} which you can configure to
 * calculate R(s), R(s,a),
 * or R(s,a,s')</li>
 * <li>How is the model punished/pleasured according to the quality of
 * {@link Territory}s that it chooses to place
 * armies at? This is done via a {@link MyPlacementRewardFunction} which you can
 * configure
 * to calculate R(s), R(s,t), or R(s,t,s')</li>
 * </ol>
 *
 */
public class RiskQAgent
        extends NeuralQAgent {

    private final double EPSILON = 1.0;
    private Random rng;

    public RiskQAgent(int agentId) {
        super(agentId);
        this.rng = new Random();
    }

    /**
     * A method to create your neural network architecture. This is done by making
     * three separate {@link Sequential}
     * instances (with appropriate dimensions) and then chucking them into the
     * {@link DualDecoderModel} class I made
     * for you which coordinates them.
     *
     * @return The {@link DualDecoderModel} which coordinates the three neural
     *         networks you make here.
     */
    public DualDecoderModel initModel() {

        // default model..you will likely want to change this

        // lookup how many features each item has
        final int hiddenTerritoryDim = 16;

        // state encoder
        Sequential encoder = new Sequential();
        encoder.add(new HackyTerritoryLinear(MyStateSensorArray.NUM_FEATURES_PER_TERRITORY, hiddenTerritoryDim));
        encoder.add(new HackyTerritoryRMSNorm());
        encoder.add(new HackyPassThroughReLU());
        encoder.add(new HackyKipfWellingGCN(hiddenTerritoryDim, hiddenTerritoryDim));
        encoder.add(new HackyTerritoryRMSNorm());
        encoder.add(new HackyPassThroughReLU());

        // action decoder
        Sequential actionDecoder = new Sequential();
        actionDecoder.add(new HackyTerritoryConcat(
                hiddenTerritoryDim,
                MyActionSensorArray.NUM_FEATURES_PER_TERRITORY));
        actionDecoder.add(new HackyTerritoryLinear(
                hiddenTerritoryDim + MyActionSensorArray.NUM_FEATURES_PER_TERRITORY,
                hiddenTerritoryDim));
        actionDecoder.add(new HackyTerritoryRMSNorm());
        actionDecoder.add(new HackyPassThroughReLU());
        actionDecoder.add(new HackyKipfWellingGCN(hiddenTerritoryDim, hiddenTerritoryDim));
        actionDecoder.add(new HackyTerritoryRMSNorm());
        actionDecoder.add(new HackyPassThroughReLU());
        actionDecoder.add(new HackyReduceSum(hiddenTerritoryDim));
        actionDecoder.add(new HackyRMSNorm());
        actionDecoder.add(new HackyPassThroughDense(hiddenTerritoryDim, 1));

        // placement decoder
        Sequential placementDecoder = new Sequential();
        placementDecoder.add(new HackyTerritoryConcat(
                hiddenTerritoryDim,
                MyPlacementSensorArray.NUM_FEATURES_PER_TERRITORY));
        placementDecoder.add(new HackyTerritoryLinear(
                hiddenTerritoryDim + MyPlacementSensorArray.NUM_FEATURES_PER_TERRITORY,
                hiddenTerritoryDim));
        placementDecoder.add(new HackyTerritoryRMSNorm());
        placementDecoder.add(new HackyPassThroughReLU());
        placementDecoder.add(new HackyKipfWellingGCN(hiddenTerritoryDim, hiddenTerritoryDim));
        placementDecoder.add(new HackyTerritoryRMSNorm());
        placementDecoder.add(new HackyPassThroughReLU());
        placementDecoder.add(new HackyReduceSum(hiddenTerritoryDim));
        placementDecoder.add(new HackyRMSNorm());
        placementDecoder.add(new HackyPassThroughDense(hiddenTerritoryDim, 1));

        var model = new DualDecoderModel(encoder, actionDecoder, placementDecoder);
        // try {
        // model.load("/Users/kawgit/Assignments/risk/qFunction10.model");
        // System.out.println("Model loaded");
        // } catch (Exception e) {
        // System.out.println("Model not found");
        // }
        return model;
    }

    /**
     * A method to create your state sensor suite.
     *
     * @return Your state sensor suite
     */
    @Override
    public StateSensorArray createStateSensors() {
        return new MyStateSensorArray(this.agentId());
    }

    /**
     * A method to create your action sensor suite.
     *
     * @return Your action sensor suite
     */
    @Override
    public ActionSensorArray createActionSensors() {
        return new MyActionSensorArray(this.agentId());
    }

    /**
     * A method to create your placement sensor suite.
     *
     * @return Your placement sensor suite
     */
    @Override
    public PlacementSensorArray createPlacementSensors() {
        return new MyPlacementSensorArray(this.agentId());
    }

    /**
     * A method to create your action reward function.
     *
     * @return Your action reward function
     */
    @Override
    public RewardFunction<Action> createActionReward() {
        return new MyActionRewardFunction(this.agentId());
    }

    /**
     * A method to create your placement reward function.
     *
     * @return Your placement reward function
     */
    @Override
    public RewardFunction<Territory> createPlacementReward() {
        return new MyPlacementRewardFunction(this.agentId());
    }

    public <T> T chooseRandom(final List<T> list) {
        return list.get(this.rng.nextInt(list.size()));
    }

    public <T> T chooseRandomWithLogits(final List<T> list, final double[] logits, final double temperature) {
        assert list.size() == logits.length || logits.length == 0
                : "List size must match logits length, or logits must be empty.";

        double maxLogit = Double.NEGATIVE_INFINITY;
        for (double logit : logits) {
            if (logit > maxLogit) {
                maxLogit = logit;
            }
        }

        double[] expLogits = new double[logits.length];
        double sum = 0.0;
        for (int i = 0; i < logits.length; i++) {
            expLogits[i] = Math.exp((logits[i] - maxLogit) / temperature);
            sum += expLogits[i];
        }

        for (int i = 0; i < logits.length; i++) {
            expLogits[i] /= sum;
        }

        System.out.println(
                "expLogits: "
                        + Arrays.toString(Arrays.stream(expLogits).map(a -> Math.round(a * 100) / 100.0).toArray()));

        double threshold = this.rng.nextDouble();
        double cumulative = 0.0;

        for (int i = 0; i < expLogits.length; i++) {
            cumulative += expLogits[i];
            if (threshold <= cumulative) {
                return list.get(i);
            }
        }

        return list.get(list.size() - 1);
    }

    @FunctionalInterface
    public interface ModelForward {
        Matrix apply(Matrix stateFeatures, Matrix itemFeatures) throws Exception;
    }

    public <T> T chooseRandomWithModelSoftmax(final GameView game,
            final List<T> options,
            final List<Matrix> featureVectors,
            final ModelForward modelCall,
            final double temperature) {
        assert !options.isEmpty() && options.size() == featureVectors.size()
                : "Options and features must be non-empty and matching in size.";

        if (options.size() == 1) {
            return options.get(0);
        }

        final Matrix stateFeatures = this.getStateFeatureVector(game);
        double[] logits = new double[options.size()];

        for (int i = 0; i < options.size(); i++) {
            try {
                // Execute the model forward pass (Action or Placement)
                logits[i] = modelCall.apply(stateFeatures, featureVectors.get(i)).item();
            } catch (Exception e) {
                logits[i] = Double.NEGATIVE_INFINITY;
                System.out.println(e.getLocalizedMessage());
            }
        }

        return chooseRandomWithLogits(options, logits, temperature);
    }

    /**
     * A method to choose an {@link Action} when it is in the redeem phase of a
     * turn. You are free to write your own
     * code to choose which move to explore however your decision should be
     * stochastic (e.g. determinism is bad).
     *
     * @param game           the current state of the game
     * @param actionCounter  how many actions you've made so far in this turn
     * @param canRedeemCards can you redeem cards
     * @return the {@link Action} to do
     */
    @Override
    public Action getExplorationRedeemAction(final GameView game,
            final int actionCounter,
            final boolean canRedeemCards) {
        System.out.println("getExplorationRedeemAction");
        final List<Action> options = this.getRedeemActions(game, actionCounter, canRedeemCards,
                game.getAgentInventory(this.agentId()).size() < 5);
        List<Matrix> features = options.stream()
                .map(action -> this.getActionFeatureVector(game, actionCounter, action))
                .toList();
        return this.chooseRandomWithModelSoftmax(game, options, features,
                this.getModel()::actionForward, 1.0);
    }

    /**
     * A method to decide whether to listen to your q-function or not. This will be
     * called ever time your agent
     * needs to decide what move to make in the redeem phase of your turn.
     *
     * @param game           the current state of the game
     * @param actionCounter  how many actions you've made so far in this turn
     * @param canRedeemCards can you redeem cards
     * @return <code>true</code> if <code>getExplorationRedeemAction</code> should
     *         be called or if your action
     *         q-function should be argmaxed.
     */
    @Override
    public boolean shouldExploreRedeemMovePhase(final GameView game,
            final int actionCounter,
            final boolean canRedeemCards) {
        return this.rng.nextDouble() < EPSILON;
    }

    /**
     * A method to choose an {@link Action} when it is in the attacking phase of a
     * turn. You are free to write your own
     * code to choose which move to explore however your decision should be
     * stochastic (e.g. determinism is bad).
     *
     * @param game           the current state of the game
     * @param actionCounter  how many actions you've made so far in this turn
     * @param canRedeemCards can you redeem cards
     * @return the {@link Action} to do
     */
    @Override
    public Action getExplorationAttackActionRedeemIfForced(final GameView game,
            final int actionCounter,
            final boolean canRedeemCards) {
        System.out.println("getExplorationAttackActionRedeemIfForced");
        final List<Action> options = this.getAttackRedeemActions(game, actionCounter, canRedeemCards);
        List<Matrix> features = options.stream()
                .map(action -> this.getActionFeatureVector(game, actionCounter, action))
                .toList();
        return this.chooseRandomWithModelSoftmax(game, options, features,
                this.getModel()::actionForward, 5.0);
    }

    /**
     * A method to decide whether to listen to your q-function or not. This will be
     * called ever time your agent
     * needs to decide what move to make in the attacking phase of your turn.
     *
     * @param game           the current state of the game
     * @param actionCounter  how many actions you've made so far in this turn
     * @param canRedeemCards can you redeem cards
     * @return <code>true</code> if
     *         <code>getExplorationAttackActionRedeemIfForced</code> should be
     *         called or
     *         if your action q-function should be argmaxed.
     */
    @Override
    public boolean shouldExploreAttackRedeemIfForcedMovePhase(final GameView game,
            final int actionCounter,
            final boolean canRedeemCards) {
        return this.rng.nextDouble() < EPSILON;
    }

    /**
     * A method to choose an {@link Action} when it is in the fortifying phase of a
     * turn. You are free to write your own
     * code to choose which move to explore however your decision should be
     * stochastic (e.g. determinism is bad).
     *
     * @param game           the current state of the game
     * @param actionCounter  how many actions you've made so far in this turn
     * @param canRedeemCards can you redeem cards
     * @return the {@link Action} to do
     */
    @Override
    public Action getExplorationFortifySkipAction(final GameView game,
            final int actionCounter,
            final boolean canRedeemCards) {
        System.out.println("getExplorationFortifySkipAction");
        final List<Action> options = this.getFortifyActions(game, actionCounter, canRedeemCards);
        List<Matrix> features = options.stream()
                .map(action -> this.getActionFeatureVector(game, actionCounter, action))
                .toList();
        return this.chooseRandomWithModelSoftmax(game, options, features,
                this.getModel()::actionForward, 1.0);
    }

    /**
     * A method to decide whether to listen to your q-function or not. This will be
     * called ever time your agent
     * needs to decide what move to make in the fortifying phase of your turn.
     *
     * @param game           the current state of the game
     * @param actionCounter  how many actions you've made so far in this turn
     * @param canRedeemCards can you redeem cards
     * @return <code>true</code> if <code>getExplorationFortifySkipAction</code>
     *         should be called or
     *         if your action q-function should be argmaxed.
     */
    @Override
    public boolean shouldExploreFortifySkipMovePhase(final GameView game,
            final int actionCounter,
            final boolean canRedeemCards) {
        return this.rng.nextDouble() < EPSILON;
    }

    /**
     * A method to choose an {@link Territory} when it is in the army placing phase
     * of a turn (or during game setup).
     * You are free to write your own code to choose which move to explore however
     * your decision should be stochastic
     * (e.g. determinism is bad).
     *
     * @param game            the current state of the game
     * @param isDuringSetup   is this during the game setup or at the beginning of
     *                        your move
     * @param remainingArmies number of armies left to place
     * @return the {@link Territory} to place an army at
     */
    @Override
    public Territory getExplorationPlacement(final GameView game,
            final boolean isDuringSetup,
            final int remainingArmies) {
        System.out.println("getExplorationPlacement");
        final List<Territory> options = this.getPotentialPlacements(game, isDuringSetup, remainingArmies);
        List<Matrix> features = options.stream()
                .map(option -> this.getPlacementFeatureVector(game, remainingArmies, option))
                .toList();
        return this.chooseRandomWithModelSoftmax(game, options, features,
                this.getModel()::placementForward, 5);

    }

    /**
     * A method to decide whether to listen to your q-function or not. This will be
     * called ever time your agent
     * needs to decide what {@link Territory} to place an army at.
     *
     * @param game            the current state of the game
     * @param isDuringSetup   is this during the game setup or at the beginning of
     *                        your move
     * @param remainingArmies number of armies left to place
     * @return <code>true</code> if <code>getExplorationPlacement</code> should be
     *         called or
     *         if your action q-function should be argmaxed.
     */
    @Override
    public boolean shouldExplorePlacementPhase(final GameView game,
            final boolean isDuringSetup,
            final int remainingArmies) {
        return this.rng.nextDouble() < EPSILON;
    }

    public class HackyTerritoryRMSNorm extends Module {
        private static final int NUM_TERRITORIES = 42;
        private final double eps = 1e-5;

        public HackyTerritoryRMSNorm() {
        }

        public Matrix forward(Matrix X) throws Exception {
            assert (X.getShape().numCols() - 1) % NUM_TERRITORIES == 0;
            int territory_dim = (X.getShape().numCols() - 1) / NUM_TERRITORIES;

            int rows = X.getShape().numRows();
            Matrix out = Matrix.zeros(rows, X.getShape().numCols());
            for (int r = 0; r < rows; r++)
                out.set(r, X.getShape().numCols() - 1, X.get(r, X.getShape().numCols() - 1));

            for (int r = 0; r < rows; r++) {
                for (int i = 0; i < NUM_TERRITORIES; i++) {
                    int offset = i * territory_dim;
                    double sumSq = 0.0;
                    for (int c = 0; c < territory_dim; c++) {
                        double val = X.get(r, offset + c);
                        sumSq += val * val;
                    }
                    double rms = Math.sqrt((sumSq / territory_dim) + eps);

                    for (int c = 0; c < territory_dim; c++) {
                        double normalized = X.get(r, offset + c) / rms;
                        out.set(r, offset + c, normalized);
                    }
                }
            }
            return out;
        }

        public Matrix backwards(Matrix X, Matrix dLoss_dModule) throws Exception {
            assert (X.getShape().numCols() - 1) % NUM_TERRITORIES == 0;
            assert dLoss_dModule.getShape().numCols() == X.getShape().numCols();

            int territory_dim = (X.getShape().numCols() - 1) / NUM_TERRITORIES;

            int rows = X.getShape().numRows();
            Matrix dLoss_dX = Matrix.zeros(rows, X.getShape().numCols());
            for (int r = 0; r < rows; r++)
                dLoss_dX.set(r, X.getShape().numCols() - 1, dLoss_dModule.get(r, X.getShape().numCols() - 1));

            for (int r = 0; r < rows; r++) {
                for (int i = 0; i < NUM_TERRITORIES; i++) {
                    int offset = i * territory_dim;

                    double sumSq = 0.0;
                    for (int c = 0; c < territory_dim; c++) {
                        double val = X.get(r, offset + c);
                        sumSq += val * val;
                    }
                    double rms = Math.sqrt((sumSq / territory_dim) + eps);
                    double inv_rms = 1.0 / rms;

                    double sum_j = 0.0;
                    for (int c = 0; c < territory_dim; c++) {
                        sum_j += dLoss_dModule.get(r, offset + c) * X.get(r, offset + c);
                    }

                    double correction_term = sum_j / (territory_dim * rms * rms * rms);

                    for (int c = 0; c < territory_dim; c++) {
                        double dx = (dLoss_dModule.get(r, offset + c) * inv_rms)
                                - (X.get(r, offset + c) * correction_term);
                        dLoss_dX.set(r, offset + c, dx);
                    }
                }
            }

            return dLoss_dX;
        }

        public List<Parameter> getParameters() {
            return new ArrayList<>(0);
        }
    }

    public class HackyTerritoryLinear extends Module {
        private static final int NUM_TERRITORIES = 42;

        private Parameter W_territory;
        private Parameter b_territory;

        private final int input_territory_size;
        private final int output_territory_size;

        public HackyTerritoryLinear(int input_territory_size, int output_territory_size) {
            this.input_territory_size = input_territory_size;
            this.output_territory_size = output_territory_size;

            Random rng = new Random();

            double bound_t = Math.sqrt(1.0 / (double) input_territory_size);
            this.W_territory = new Parameter(Matrix.zeros(input_territory_size, output_territory_size));
            this.b_territory = new Parameter(Matrix.zeros(1, output_territory_size));
            initUniform(this.W_territory.getValue(), -bound_t, bound_t, rng);
            initUniform(this.b_territory.getValue(), -bound_t, bound_t, rng);
        }

        private void initUniform(Matrix m, double min, double max, Random rng) {
            int rows = m.getShape().numRows();
            int cols = m.getShape().numCols();
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    m.set(r, c, min + (max - min) * rng.nextDouble());
                }
            }
        }

        public Matrix forward(Matrix X) throws Exception {
            assert X.getShape().numCols() == NUM_TERRITORIES * input_territory_size + 1;
            int rows = X.getShape().numRows();

            // Sanitize inputs from sensors
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < X.getShape().numCols(); c++) {
                    double val = X.get(r, c);
                    if (Double.isNaN(val) || Double.isInfinite(val)) {
                        X.set(r, c, 0.0);
                    }
                }
            }

            Matrix territory_output = Matrix.zeros(rows, NUM_TERRITORIES * output_territory_size + 1);
            for (int r = 0; r < rows; r++)
                territory_output.set(r, NUM_TERRITORIES * output_territory_size, X.get(r, X.getShape().numCols() - 1));

            for (int i = 0; i < NUM_TERRITORIES; i++) {
                Matrix slice = X.getSlice(0, rows, i * input_territory_size, (i + 1) * input_territory_size);
                Matrix processed = slice.matmul(W_territory.getValue());
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < output_territory_size; c++) {
                        territory_output.set(r, i * output_territory_size + c,
                                processed.get(r, c) + b_territory.getValue().get(0, c));
                    }
                }
            }

            return territory_output;
        }

        public Matrix backwards(Matrix X, Matrix dLoss_dModule) throws Exception {
            assert X.getShape().numCols() == NUM_TERRITORIES * input_territory_size + 1;
            assert dLoss_dModule.getShape().numCols() == NUM_TERRITORIES * output_territory_size + 1;
            int rows = X.getShape().numRows();
            Matrix dLoss_dX = Matrix.zeros(rows, X.getShape().numCols());
            for (int r = 0; r < rows; r++)
                dLoss_dX.set(r, X.getShape().numCols() - 1,
                        dLoss_dModule.get(r, dLoss_dModule.getShape().numCols() - 1));

            Matrix gradW = this.W_territory.getGradient();
            Matrix gradB = this.b_territory.getGradient();

            for (int i = 0; i < NUM_TERRITORIES; i++) {
                Matrix dLoss_dSliceOut = dLoss_dModule.getSlice(0, rows, i * output_territory_size,
                        (i + 1) * output_territory_size);
                Matrix slice_in = X.getSlice(0, rows, i * input_territory_size, (i + 1) * input_territory_size);

                Matrix dW = slice_in.transpose().matmul(dLoss_dSliceOut);
                for (int r = 0; r < gradW.getShape().numRows(); r++) {
                    for (int c = 0; c < gradW.getShape().numCols(); c++) {
                        gradW.set(r, c, gradW.get(r, c) + dW.get(r, c) + 1e-12);
                    }
                }

                Matrix dB = dLoss_dSliceOut.sum(0);
                for (int c = 0; c < gradB.getShape().numCols(); c++) {
                    gradB.set(0, c, gradB.get(0, c) + dB.get(0, c) + 1e-12);
                }

                Matrix dLoss_dSliceIn = dLoss_dSliceOut.matmul(this.W_territory.getValue().transpose());
                dLoss_dX.copySlice(0, rows, i * input_territory_size, (i + 1) * input_territory_size, dLoss_dSliceIn);
            }

            return dLoss_dX;
        }

        public List<Parameter> getParameters() {
            List<Parameter> params = new ArrayList<>(2);
            params.add(W_territory);
            params.add(b_territory);
            return params;
        }
    }

    public class HackyTerritoryConcat extends Module {
        private static final int NUM_TERRITORIES = 42;

        private final int t1_size;
        private final int t2_size;

        public HackyTerritoryConcat(int t1_size, int t2_size) {
            this.t1_size = t1_size;
            this.t2_size = t2_size;
        }

        public Matrix forward(Matrix X) throws Exception {
            assert X.getShape().numCols() == NUM_TERRITORIES * t1_size + 1 + NUM_TERRITORIES * t2_size + 1;
            int rows = X.getShape().numRows();
            int cols = X.getShape().numCols();

            // Sanitize inputs from sensors
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    double val = X.get(r, c);
                    if (Double.isNaN(val) || Double.isInfinite(val)) {
                        X.set(r, c, 0.0);
                    }
                }
            }

            int out_t_size = t1_size + t2_size;
            Matrix out = Matrix.zeros(rows, NUM_TERRITORIES * out_t_size + 1);

            int offset_B = NUM_TERRITORIES * t1_size + 1;

            for (int i = 0; i < NUM_TERRITORIES; i++) {
                Matrix t_a = X.getSlice(0, rows, i * t1_size, (i + 1) * t1_size);
                Matrix t_b = X.getSlice(0, rows, offset_B + (i * t2_size), offset_B + ((i + 1) * t2_size));

                out.copySlice(0, rows, i * out_t_size, (i * out_t_size) + t1_size, t_a);
                out.copySlice(0, rows, (i * out_t_size) + t1_size, (i + 1) * out_t_size, t_b);
            }

            for (int r = 0; r < rows; r++) {
                double bias1 = X.get(r, NUM_TERRITORIES * t1_size);
                double bias2 = X.get(r, cols - 1);
                out.set(r, NUM_TERRITORIES * out_t_size, bias1 + bias2);
            }

            return out;
        }

        public Matrix backwards(Matrix X, Matrix dLoss_dModule) throws Exception {
            assert X.getShape().numCols() == NUM_TERRITORIES * t1_size + 1 + NUM_TERRITORIES * t2_size + 1;
            assert dLoss_dModule.getShape().numCols() == NUM_TERRITORIES * (t1_size + t2_size) + 1;
            int rows = X.getShape().numRows();
            int cols = X.getShape().numCols();
            Matrix dLoss_dX = Matrix.zeros(rows, cols);

            int offset_B = NUM_TERRITORIES * t1_size + 1;
            int out_t_size = t1_size + t2_size;

            for (int r = 0; r < rows; r++) {
                double dbias = dLoss_dModule.get(r, dLoss_dModule.getShape().numCols() - 1);
                dLoss_dX.set(r, NUM_TERRITORIES * t1_size, dbias);
                dLoss_dX.set(r, cols - 1, dbias);
            }

            for (int i = 0; i < NUM_TERRITORIES; i++) {
                Matrix dLoss_t_ab = dLoss_dModule.getSlice(0, rows, i * out_t_size, (i + 1) * out_t_size);
                Matrix dLoss_t_a = dLoss_t_ab.getSlice(0, rows, 0, t1_size);
                Matrix dLoss_t_b = dLoss_t_ab.getSlice(0, rows, t1_size, out_t_size);

                dLoss_dX.copySlice(0, rows, i * t1_size, (i + 1) * t1_size, dLoss_t_a);
                dLoss_dX.copySlice(0, rows, offset_B + (i * t2_size), offset_B + ((i + 1) * t2_size), dLoss_t_b);
            }

            return dLoss_dX;
        }

        public List<Parameter> getParameters() {
            return new ArrayList<>(0);
        }
    }

    public class HackyRMSNorm extends Module {
        private final double eps = 1e-5;

        public HackyRMSNorm() {
        }

        public Matrix forward(Matrix X) throws Exception {
            assert X.getShape().numCols() >= 2;
            int rows = X.getShape().numRows();
            int cols = X.getShape().numCols();
            int dim = cols - 1;
            Matrix out = Matrix.zeros(rows, cols);

            for (int r = 0; r < rows; r++) {
                out.set(r, cols - 1, X.get(r, cols - 1));
                double sumSq = 0.0;
                for (int c = 0; c < dim; c++) {
                    double val = X.get(r, c);
                    sumSq += val * val;
                }
                double rms = Math.sqrt((sumSq / dim) + eps);

                for (int c = 0; c < dim; c++) {
                    double normalized = X.get(r, c) / rms;
                    out.set(r, c, normalized);
                }
            }
            return out;
        }

        public Matrix backwards(Matrix X, Matrix dLoss_dModule) throws Exception {
            assert X.getShape().numCols() >= 2;
            assert dLoss_dModule.getShape().numCols() == X.getShape().numCols();
            int rows = X.getShape().numRows();
            int cols = X.getShape().numCols();
            int dim = cols - 1;
            Matrix dLoss_dX = Matrix.zeros(rows, cols);

            for (int r = 0; r < rows; r++) {
                dLoss_dX.set(r, cols - 1, dLoss_dModule.get(r, cols - 1));
                double sumSq = 0.0;
                for (int c = 0; c < dim; c++) {
                    double val = X.get(r, c);
                    sumSq += val * val;
                }
                double rms = Math.sqrt((sumSq / dim) + eps);
                double inv_rms = 1.0 / rms;

                double sum_j = 0.0;
                for (int c = 0; c < dim; c++) {
                    sum_j += dLoss_dModule.get(r, c) * X.get(r, c);
                }

                double correction_term = sum_j / (dim * rms * rms * rms);

                for (int c = 0; c < dim; c++) {
                    double dx = (dLoss_dModule.get(r, c) * inv_rms) - (X.get(r, c) * correction_term);
                    dLoss_dX.set(r, c, dx);
                }
            }

            return dLoss_dX;
        }

        public List<Parameter> getParameters() {
            return new ArrayList<>(0);
        }
    }

    public class HackyReduceSum extends Module {
        private final int vectorDim;

        public HackyReduceSum(int vectorDim) {
            this.vectorDim = vectorDim;
        }

        public Matrix forward(Matrix X) throws Exception {
            assert (X.getShape().numCols() - 1) % vectorDim == 0;
            int rows = X.getShape().numRows();
            int cols = X.getShape().numCols();
            int numVectors = (cols - 1) / vectorDim;

            Matrix out = Matrix.zeros(rows, vectorDim + 1);

            for (int r = 0; r < rows; r++) {
                out.set(r, vectorDim, X.get(r, cols - 1));
            }

            for (int i = 0; i < numVectors; i++) {
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < vectorDim; c++) {
                        out.set(r, c, out.get(r, c) + X.get(r, i * vectorDim + c));
                    }
                }
            }

            return out;
        }

        public Matrix backwards(Matrix X, Matrix dLoss_dModule) throws Exception {
            assert (X.getShape().numCols() - 1) % vectorDim == 0;
            assert dLoss_dModule.getShape().numCols() == vectorDim + 1;
            int rows = X.getShape().numRows();
            int cols = X.getShape().numCols();
            int numVectors = (cols - 1) / vectorDim;

            Matrix dLoss_dX = Matrix.zeros(rows, cols);
            for (int r = 0; r < rows; r++) {
                dLoss_dX.set(r, cols - 1, dLoss_dModule.get(r, vectorDim));
            }

            Matrix dLoss_dModule_features = dLoss_dModule.getSlice(0, rows, 0, vectorDim);
            for (int i = 0; i < numVectors; i++) {
                dLoss_dX.copySlice(0, rows, i * vectorDim, (i + 1) * vectorDim, dLoss_dModule_features);
            }

            return dLoss_dX;
        }

        public List<Parameter> getParameters() {
            return new ArrayList<>(0);
        }
    }

    public class HackyKipfWellingGCN extends Module {
        private static final int NUM_TERRITORIES = 42;

        public static final int[][] ADJACENCY_MATRIX = new int[][] {
                { 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 1, 0, 0, 0, 0 },
                { 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0,
                        1, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0,
                        1, 0, 0,
                        1, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1,
                        1, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                        1, 0, 0,
                        0, 0, 0, 0, 1, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                        1, 1, 0,
                        1, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        1, 1, 1,
                        1, 1, 0, 1, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 1, 1,
                        0, 0, 0, 1, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                        1, 1, 0,
                        1, 1, 1, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 1, 0,
                        1, 1, 1, 1, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0,
                        1, 1, 1, 1, 0, 0, 0, 0 },
                { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 1, 1,
                        0, 1, 1, 1, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                        0, 0, 0,
                        0, 0, 0, 0, 1, 1, 1, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 1, 1, 1, 1 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 1, 1, 1, 1 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0, 0, 0, 1, 1, 1 } };

        private double[][] A_hat;

        private Parameter W;
        private Parameter b;

        private final int in_features;
        private final int out_features;

        public HackyKipfWellingGCN(int in_features, int out_features) {
            this.in_features = in_features;
            this.out_features = out_features;

            Random rng = new Random();

            double bound = Math.sqrt(1.0 / (double) in_features);
            this.W = new Parameter(Matrix.zeros(in_features, out_features));
            this.b = new Parameter(Matrix.zeros(1, out_features));
            initUniform(this.W.getValue(), -bound, bound, rng);
            initUniform(this.b.getValue(), -bound, bound, rng);

            initGraphMatrix();
        }

        private void initGraphMatrix() {
            this.A_hat = new double[NUM_TERRITORIES][NUM_TERRITORIES];
            double[] D_tilde = new double[NUM_TERRITORIES];

            for (int i = 0; i < NUM_TERRITORIES; i++) {
                double degree = 0.0;
                for (int j = 0; j < NUM_TERRITORIES; j++) {
                    degree += ADJACENCY_MATRIX[i][j];
                }
                D_tilde[i] = degree;
            }

            for (int i = 0; i < NUM_TERRITORIES; i++) {
                for (int j = 0; j < NUM_TERRITORIES; j++) {
                    if (ADJACENCY_MATRIX[i][j] > 0) {
                        this.A_hat[i][j] = ADJACENCY_MATRIX[i][j] / Math.sqrt(D_tilde[i] * D_tilde[j]);
                    }
                }
            }
        }

        private void initUniform(Matrix m, double min, double max, Random rng) {
            int rows = m.getShape().numRows();
            int cols = m.getShape().numCols();
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    m.set(r, c, min + (max - min) * rng.nextDouble());
                }
            }
        }

        private void addScaled(Matrix dest, Matrix src, double scalar) {
            int rows = dest.getShape().numRows();
            int cols = dest.getShape().numCols();
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    dest.set(r, c, dest.get(r, c) + (src.get(r, c) * scalar));
                }
            }
        }

        public Matrix forward(Matrix X) throws Exception {
            assert X.getShape().numCols() == NUM_TERRITORIES * in_features + 1;
            int rows = X.getShape().numRows();
            Matrix[] H = new Matrix[NUM_TERRITORIES];

            for (int i = 0; i < NUM_TERRITORIES; i++) {
                Matrix slice = X.getSlice(0, rows, i * in_features, (i + 1) * in_features);
                H[i] = slice.matmul(W.getValue());
            }

            int cols = X.getShape().numCols();
            Matrix out = Matrix.zeros(rows, NUM_TERRITORIES * out_features + 1);

            for (int i = 0; i < NUM_TERRITORIES; i++) {
                Matrix aggregated = Matrix.zeros(rows, out_features);
                for (int j = 0; j < NUM_TERRITORIES; j++) {
                    if (A_hat[i][j] != 0.0) {
                        addScaled(aggregated, H[j], A_hat[i][j]);
                    }
                }
                for (int r = 0; r < rows; r++) {
                    for (int c = 0; c < out_features; c++) {
                        aggregated.set(r, c, aggregated.get(r, c) + b.getValue().get(0, c));
                    }
                }
                out.copySlice(0, rows, i * out_features, (i + 1) * out_features, aggregated);
            }

            for (int r = 0; r < rows; r++) {
                out.set(r, NUM_TERRITORIES * out_features, X.get(r, cols - 1));
            }

            return out;
        }

        public Matrix backwards(Matrix X, Matrix dLoss_dModule) throws Exception {
            assert X.getShape().numCols() == NUM_TERRITORIES * in_features + 1;
            assert dLoss_dModule.getShape().numCols() == NUM_TERRITORIES * out_features + 1;
            int rows = X.getShape().numRows();
            Matrix dLoss_dX = Matrix.zeros(rows, NUM_TERRITORIES * in_features + 1);
            for (int r = 0; r < rows; r++) {
                dLoss_dX.set(r, NUM_TERRITORIES * in_features, dLoss_dModule.get(r, NUM_TERRITORIES * out_features));
            }

            Matrix[] dOut = new Matrix[NUM_TERRITORIES];
            for (int i = 0; i < NUM_TERRITORIES; i++) {
                dOut[i] = dLoss_dModule.getSlice(0, rows, i * out_features, (i + 1) * out_features);
            }

            Matrix[] dH = new Matrix[NUM_TERRITORIES];
            for (int i = 0; i < NUM_TERRITORIES; i++) {
                dH[i] = Matrix.zeros(rows, out_features);
                for (int j = 0; j < NUM_TERRITORIES; j++) {
                    if (A_hat[i][j] != 0.0) {
                        addScaled(dH[i], dOut[j], A_hat[i][j]);
                    }
                }
            }

            Matrix gradW = this.W.getGradient();
            Matrix gradB = this.b.getGradient();

            for (int i = 0; i < NUM_TERRITORIES; i++) {
                Matrix slice_in = X.getSlice(0, rows, i * in_features, (i + 1) * in_features);

                Matrix dW = slice_in.transpose().matmul(dH[i]);
                for (int r = 0; r < gradW.getShape().numRows(); r++) {
                    for (int c = 0; c < gradW.getShape().numCols(); c++) {
                        gradW.set(r, c, gradW.get(r, c) + dW.get(r, c) + 1e-12);
                    }
                }

                Matrix dB = dOut[i].sum(0);
                for (int c = 0; c < gradB.getShape().numCols(); c++) {
                    gradB.set(0, c, gradB.get(0, c) + dB.get(0, c) + 1e-12);
                }

                Matrix dLoss_dSliceIn = dH[i].matmul(this.W.getValue().transpose());
                dLoss_dX.copySlice(0, rows, i * in_features, (i + 1) * in_features, dLoss_dSliceIn);
            }

            return dLoss_dX;
        }

        public List<Parameter> getParameters() {
            List<Parameter> params = new ArrayList<>(2);
            params.add(W);
            params.add(b);
            return params;
        }
    }

    public class HackyPassThroughReLU extends Module {
        public HackyPassThroughReLU() {
        }

        public Matrix forward(Matrix X) throws Exception {
            assert X.getShape().numCols() >= 2;
            int rows = X.getShape().numRows();
            int cols = X.getShape().numCols();
            Matrix out = Matrix.zeros(rows, cols);
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols - 1; c++) {
                    out.set(r, c, Math.max(0.0, X.get(r, c)));
                }
                out.set(r, cols - 1, X.get(r, cols - 1));
            }
            return out;
        }

        public Matrix backwards(Matrix X, Matrix dLoss_dModule) throws Exception {
            assert X.getShape().numCols() >= 2;
            assert dLoss_dModule.getShape().numCols() == X.getShape().numCols();
            int rows = X.getShape().numRows();
            int cols = X.getShape().numCols();
            Matrix dLoss_dX = Matrix.zeros(rows, cols);
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols - 1; c++) {
                    if (X.get(r, c) > 0) {
                        dLoss_dX.set(r, c, dLoss_dModule.get(r, c));
                    }
                }
                dLoss_dX.set(r, cols - 1, dLoss_dModule.get(r, cols - 1));
            }
            return dLoss_dX;
        }

        public List<Parameter> getParameters() {
            return new ArrayList<>(0);
        }
    }

    public class HackyPassThroughDense extends Module {
        private Parameter W;
        private Parameter b;
        private final int in_features;
        private final int out_features;

        private int numForwardPasses;
        private int numBackwardsPasses;

        public HackyPassThroughDense(int in_features, int out_features) {
            this.in_features = in_features;
            this.out_features = out_features;

            Random rng = new Random();
            double bound = Math.sqrt(1.0 / (double) in_features);
            this.W = new Parameter(Matrix.zeros(in_features + 1, out_features));
            this.b = new Parameter(Matrix.zeros(1, out_features));

            // Initialize normal features
            for (int r = 0; r < in_features; r++) {
                for (int c = 0; c < out_features; c++) {
                    this.W.getValue().set(r, c, -bound + (2 * bound) * rng.nextDouble());
                }
            }

            // Initialize bias scaling to 100.0
            for (int c = 0; c < out_features; c++) {
                this.W.getValue().set(in_features, c, 150.0);
            }

            this.numForwardPasses = 0;
            this.numBackwardsPasses = 0;

            initUniform(this.b.getValue(), -bound, bound, rng);
        }

        private void initUniform(Matrix m, double min, double max, Random rng) {
            int rows = m.getShape().numRows();
            int cols = m.getShape().numCols();
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    m.set(r, c, min + (max - min) * rng.nextDouble());
                }
            }
        }

        public Matrix forward(Matrix X) throws Exception {
            this.numForwardPasses++;

            assert X.getShape().numCols() == in_features + 1;
            int rows = X.getShape().numRows();
            Matrix out = X.matmul(W.getValue());
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < out_features; c++) {
                    out.set(r, c, out.get(r, c) + b.getValue().get(0, c));
                }
            }
            return out;
        }

        public Matrix backwards(Matrix X, Matrix dLoss_dModule) throws Exception {
            this.numBackwardsPasses++;

            if (Math.random() * 100 < 1) {
                double networkContribution = 0.0;
                for (int c = 0; c < in_features; c++) {
                    networkContribution += X.get(0, c) * W.getValue().get(c, 0);
                }
                networkContribution += b.getValue().get(0, 0);

                double biasContribution = X.get(0, in_features) *
                        W.getValue().get(in_features, 0);
                double output0 = networkContribution + biasContribution;
                // MSE loss in the framework computes dL/dy = (y_pred - y_true) / N
                // So y_true = y_pred - dL/dy * N
                double estimatedGroundTruth = output0 - (dLoss_dModule.get(0, 0) *
                        X.getShape().numRows());

                System.out.println(" --- --- --- --- --- --- ");
                System.out.println("numForwardPasses: " + numForwardPasses);
                System.out.println("numBackwardsPasses: " + numBackwardsPasses);
                System.out.println("bias: " + X.get(0, in_features));
                System.out.println("networkContribution: " + networkContribution);
                System.out.println("biasWeight: " + W.getValue().get(in_features, 0));
                System.out.println("biasContribution: " + biasContribution);
                System.out.println("output: " + output0);
                System.out.println("estimatedGroundTruth: " + estimatedGroundTruth);
            }

            assert X.getShape().numCols() == in_features + 1;
            assert dLoss_dModule.getShape().numCols() == out_features;

            // Sanitize incoming loss gradients from the framework
            for (int r = 0; r < dLoss_dModule.getShape().numRows(); r++) {
                for (int c = 0; c < dLoss_dModule.getShape().numCols(); c++) {
                    double val = dLoss_dModule.get(r, c);
                    if (Double.isNaN(val) || Double.isInfinite(val) || Math.abs(val) > 10000.0) {
                        dLoss_dModule.set(r, c, 0.0);
                    }
                }
            }

            Matrix dW = X.transpose().matmul(dLoss_dModule);
            Matrix gradW = W.getGradient();
            for (int r = 0; r < gradW.getShape().numRows(); r++) {
                for (int c = 0; c < gradW.getShape().numCols(); c++) {
                    gradW.set(r, c, gradW.get(r, c) + dW.get(r, c) + 1e-12);
                }
            }

            Matrix dB = dLoss_dModule.sum(0);
            Matrix gradB = b.getGradient();
            for (int c = 0; c < gradB.getShape().numCols(); c++) {
                gradB.set(0, c, gradB.get(0, c) + dB.get(0, c) + 1e-12);
            }

            return dLoss_dModule.matmul(W.getValue().transpose());
        }

        public List<Parameter> getParameters() {
            List<Parameter> params = new ArrayList<>(2);
            params.add(W);
            params.add(b);
            return params;
        }
    }

}
