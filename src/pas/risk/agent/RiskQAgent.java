package pas.risk.agent;

// SYSTEM IMPORTS
import java.util.*;

// JAVA PROJECT IMPORTS
import edu.bu.jmat.Matrix;
import edu.bu.jnn.layers.*;
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

    public RiskQAgent(int agentId) {
        super(agentId);
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
        encoder.add(new ReLU());
        encoder.add(new HackyKipfWellingGCN(hiddenTerritoryDim, hiddenTerritoryDim));
        encoder.add(new HackyTerritoryRMSNorm());
        encoder.add(new ReLU());

        // action decoder
        Sequential actionDecoder = new Sequential();
        actionDecoder.add(new HackyTerritoryConcat(
                hiddenTerritoryDim,
                MyActionSensorArray.NUM_FEATURES_PER_TERRITORY));
        actionDecoder.add(new HackyTerritoryLinear(
                hiddenTerritoryDim + MyActionSensorArray.NUM_FEATURES_PER_TERRITORY,
                hiddenTerritoryDim));
        actionDecoder.add(new HackyTerritoryRMSNorm());
        actionDecoder.add(new ReLU());
        actionDecoder.add(new HackyKipfWellingGCN(hiddenTerritoryDim, hiddenTerritoryDim));
        actionDecoder.add(new HackyTerritoryRMSNorm());
        actionDecoder.add(new ReLU());
        actionDecoder.add(new HackyReduceSum(hiddenTerritoryDim));
        actionDecoder.add(new HackyRMSNorm());
        actionDecoder.add(new Dense(hiddenTerritoryDim, 1));

        // placement decoder
        Sequential placementDecoder = new Sequential();
        placementDecoder.add(new HackyTerritoryConcat(
                hiddenTerritoryDim,
                MyPlacementSensorArray.NUM_FEATURES_PER_TERRITORY));
        placementDecoder.add(new HackyTerritoryLinear(
                hiddenTerritoryDim + MyPlacementSensorArray.NUM_FEATURES_PER_TERRITORY,
                hiddenTerritoryDim));
        placementDecoder.add(new HackyTerritoryRMSNorm());
        placementDecoder.add(new ReLU());
        placementDecoder.add(new HackyKipfWellingGCN(hiddenTerritoryDim, hiddenTerritoryDim));
        placementDecoder.add(new HackyTerritoryRMSNorm());
        placementDecoder.add(new ReLU());
        placementDecoder.add(new HackyReduceSum(hiddenTerritoryDim));
        placementDecoder.add(new HackyRMSNorm());
        placementDecoder.add(new Dense(hiddenTerritoryDim, 1));

        return new DualDecoderModel(encoder, actionDecoder, placementDecoder);
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

    public static <T> T chooseRandom(final List<T> list,
            final Random random) {
        return list.get(random.nextInt(list.size()));
    }

    public static <T> T chooseRandomWithLogits(final List<T> list, final double[] logits) {
        return chooseRandomWithLogits(list, logits, 5);
    }

    public static <T> T chooseRandomWithLogits(final List<T> list, final double[] logits, final double temperature) {
        if (list.size() != logits.length || logits.length == 0) {
            throw new IllegalArgumentException();
        }

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

        double maxProb = 0;
        for (int i = 0; i < logits.length; i++) {
            expLogits[i] /= sum;
            if (expLogits[i] > maxProb) {
                maxProb = expLogits[i];
            }
        }

        System.out.println(list.size() + " " + maxProb);

        double threshold = new Random().nextDouble();
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
            final ModelForward modelCall) {
        if (options.isEmpty() || options.size() != featureVectors.size()) {
            throw new IllegalArgumentException("Options and features must be non-empty and matching in size.");
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

        return chooseRandomWithLogits(options, logits);
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
        final List<Action> options = this.getRedeemActions(game, actionCounter, canRedeemCards,
                game.getAgentInventory(this.agentId()).size() < 5);
        List<Matrix> features = options.stream()
                .map(action -> this.getActionFeatureVector(game, actionCounter, action))
                .toList();
        return this.chooseRandomWithModelSoftmax(game, options, features, this.getModel()::actionForward);
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
        return true;
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
        final List<Action> options = this.getAttackRedeemActions(game, actionCounter, canRedeemCards);
        List<Matrix> features = options.stream()
                .map(action -> this.getActionFeatureVector(game, actionCounter, action))
                .toList();
        return this.chooseRandomWithModelSoftmax(game, options, features, this.getModel()::actionForward);
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
        return true;
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
        final List<Action> options = this.getFortifyActions(game, actionCounter, canRedeemCards);
        List<Matrix> features = options.stream()
                .map(action -> this.getActionFeatureVector(game, actionCounter, action))
                .toList();
        return this.chooseRandomWithModelSoftmax(game, options, features, this.getModel()::actionForward);
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
        return true;
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
        final List<Territory> options = this.getPotentialPlacements(game, isDuringSetup, remainingArmies);
        List<Matrix> features = options.stream()
                .map(option -> this.getPlacementFeatureVector(game, remainingArmies, option))
                .toList();
        return this.chooseRandomWithModelSoftmax(game, options, features, this.getModel()::placementForward);

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
        return true;
    }

    public class HackyTerritoryRMSNorm extends Module {
        private static final int NUM_TERRITORIES = 42;
        private final double eps = 1e-8;

        public HackyTerritoryRMSNorm() {
        }

        public Matrix forward(Matrix X) throws Exception {
            int territory_dim = X.getShape().numCols() / NUM_TERRITORIES;

            int rows = X.getShape().numRows();
            Matrix out = Matrix.zeros(rows, NUM_TERRITORIES * territory_dim);

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

            int territory_dim = X.getShape().numCols() / NUM_TERRITORIES;

            int rows = X.getShape().numRows();
            Matrix dLoss_dX = Matrix.zeros(rows, NUM_TERRITORIES * territory_dim);

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
            Matrix territory_output = null;
            int rows = X.getShape().numRows();

            for (int i = 0; i < NUM_TERRITORIES; i++) {
                Matrix slice = X.getSlice(0, rows, i * input_territory_size, (i + 1) * input_territory_size);
                Matrix processed = slice.matmul(W_territory.getValue()).add(b_territory.getValue());
                territory_output = (territory_output == null) ? processed : concatCols(territory_output, processed);
            }

            return territory_output;
        }

        public Matrix backwards(Matrix X, Matrix dLoss_dModule) throws Exception {
            int rows = X.getShape().numRows();
            Matrix dLoss_dX = Matrix.zeros(rows, X.getShape().numCols());

            for (int i = 0; i < NUM_TERRITORIES; i++) {
                Matrix dLoss_dSliceOut = dLoss_dModule.getSlice(0, rows, i * output_territory_size,
                        (i + 1) * output_territory_size);
                Matrix slice_in = X.getSlice(0, rows, i * input_territory_size, (i + 1) * input_territory_size);

                this.W_territory
                        .setGradient(this.W_territory.getGradient().add(slice_in.transpose().matmul(dLoss_dSliceOut)));
                this.b_territory.setGradient(this.b_territory.getGradient().add(dLoss_dSliceOut.sum(0)));

                Matrix dLoss_dSliceIn = dLoss_dSliceOut.matmul(this.W_territory.getValue().transpose());
                dLoss_dX.copySlice(0, rows, i * input_territory_size, (i + 1) * input_territory_size, dLoss_dSliceIn);
            }

            return dLoss_dX;
        }

        private Matrix concatCols(Matrix A, Matrix B) {
            int rows = A.getShape().numRows();
            int colsA = A.getShape().numCols();
            int colsB = B.getShape().numCols();
            Matrix out = Matrix.zeros(rows, colsA + colsB);
            out.copySlice(0, rows, 0, colsA, A);
            out.copySlice(0, rows, colsA, colsA + colsB, B);
            return out;
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
            int rows = X.getShape().numRows();
            int out_t_size = t1_size + t2_size;
            Matrix out = Matrix.zeros(rows, NUM_TERRITORIES * out_t_size);

            int offset_B = NUM_TERRITORIES * t1_size;

            for (int i = 0; i < NUM_TERRITORIES; i++) {
                Matrix t_a = X.getSlice(0, rows, i * t1_size, (i + 1) * t1_size);
                Matrix t_b = X.getSlice(0, rows, offset_B + (i * t2_size), offset_B + ((i + 1) * t2_size));

                out.copySlice(0, rows, i * out_t_size, (i * out_t_size) + t1_size, t_a);
                out.copySlice(0, rows, (i * out_t_size) + t1_size, (i + 1) * out_t_size, t_b);
            }

            return out;
        }

        public Matrix backwards(Matrix X, Matrix dLoss_dModule) throws Exception {
            int rows = X.getShape().numRows();
            int cols = X.getShape().numCols();
            Matrix dLoss_dX = Matrix.zeros(rows, cols);

            int offset_B = NUM_TERRITORIES * t1_size;
            int out_t_size = t1_size + t2_size;

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
        private final double eps = 1e-8;

        public HackyRMSNorm() {
        }

        public Matrix forward(Matrix X) throws Exception {
            int rows = X.getShape().numRows();
            int dim = X.getShape().numCols();
            Matrix out = Matrix.zeros(rows, dim);

            for (int r = 0; r < rows; r++) {
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
            int rows = X.getShape().numRows();
            int dim = X.getShape().numCols();
            Matrix dLoss_dX = Matrix.zeros(rows, dim);

            for (int r = 0; r < rows; r++) {
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
            int rows = X.getShape().numRows();
            int cols = X.getShape().numCols();
            int numVectors = cols / vectorDim;

            Matrix out = Matrix.zeros(rows, vectorDim);

            for (int i = 0; i < numVectors; i++) {
                Matrix slice = X.getSlice(0, rows, i * vectorDim, (i + 1) * vectorDim);
                out = out.add(slice);
            }

            return out;
        }

        public Matrix backwards(Matrix X, Matrix dLoss_dModule) throws Exception {
            int rows = X.getShape().numRows();
            int cols = X.getShape().numCols();
            int numVectors = cols / vectorDim;

            Matrix dLoss_dX = Matrix.zeros(rows, cols);

            for (int i = 0; i < numVectors; i++) {
                dLoss_dX.copySlice(0, rows, i * vectorDim, (i + 1) * vectorDim, dLoss_dModule);
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
            int[][] A_tilde = new int[NUM_TERRITORIES][NUM_TERRITORIES];

            boolean hasAdjacency = ADJACENCY_MATRIX != null && ADJACENCY_MATRIX.length == NUM_TERRITORIES;

            for (int i = 0; i < NUM_TERRITORIES; i++) {
                double degree = 0.0;
                for (int j = 0; j < NUM_TERRITORIES; j++) {
                    A_tilde[i][j] = hasAdjacency ? ADJACENCY_MATRIX[i][j] : 0;
                    if (i == j) {
                        A_tilde[i][j] += 1;
                    }
                    degree += A_tilde[i][j];
                }
                D_tilde[i] = degree;
            }

            for (int i = 0; i < NUM_TERRITORIES; i++) {
                for (int j = 0; j < NUM_TERRITORIES; j++) {
                    if (A_tilde[i][j] > 0) {
                        this.A_hat[i][j] = A_tilde[i][j] / Math.sqrt(D_tilde[i] * D_tilde[j]);
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
            // System.out.println("Forwards");

            int rows = X.getShape().numRows();
            Matrix[] H = new Matrix[NUM_TERRITORIES];

            for (int i = 0; i < NUM_TERRITORIES; i++) {
                Matrix slice = X.getSlice(0, rows, i * in_features, (i + 1) * in_features);
                H[i] = slice.matmul(W.getValue()).add(b.getValue());
            }

            Matrix out = Matrix.zeros(rows, NUM_TERRITORIES * out_features);

            for (int i = 0; i < NUM_TERRITORIES; i++) {
                Matrix aggregated = Matrix.zeros(rows, out_features);
                for (int j = 0; j < NUM_TERRITORIES; j++) {
                    if (A_hat[i][j] != 0.0) {
                        addScaled(aggregated, H[j], A_hat[i][j]);
                    }
                }
                out.copySlice(0, rows, i * out_features, (i + 1) * out_features, aggregated);
            }

            return out;
        }

        public Matrix backwards(Matrix X, Matrix dLoss_dModule) throws Exception {
            System.out.println("Backwards");

            int rows = X.getShape().numRows();
            Matrix dLoss_dX = Matrix.zeros(rows, NUM_TERRITORIES * in_features);

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

            for (int i = 0; i < NUM_TERRITORIES; i++) {
                Matrix slice_in = X.getSlice(0, rows, i * in_features, (i + 1) * in_features);

                this.W.setGradient(this.W.getGradient().add(slice_in.transpose().matmul(dH[i])));
                this.b.setGradient(this.b.getGradient().add(dH[i].sum(0)));

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
}
