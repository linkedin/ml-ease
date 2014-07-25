
package de.bwaldvogel.liblinear;

import static de.bwaldvogel.liblinear.Linear.atof;
import static de.bwaldvogel.liblinear.Linear.atoi;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.StringTokenizer;


public class Train {

    public static void main(String[] args) throws IOException, InvalidInputDataException {
        new Train().run(args);
    }

    private double    bias             = 1;
    private boolean   cross_validation = false;
    private String    inputFilename;
    private String    modelFilename;
    private int       nr_fold;
    private Parameter param            = null;
    private Problem   prob             = null;

    private void do_cross_validation() {
        int[] target = new int[prob.l];

        long start, stop;
        start = System.currentTimeMillis();
        Linear.crossValidation(prob, param, nr_fold, target);
        stop = System.currentTimeMillis();
        System.out.println("time: " + (stop - start) + " ms");

        int total_correct = 0;
        for (int i = 0; i < prob.l; i++)
            if (target[i] == prob.y[i]) ++total_correct;

        System.out.printf("correct: %d%n", total_correct);
        System.out.printf("Cross Validation Accuracy = %g%%%n", 100.0 * total_correct / prob.l);
    }

    private void exit_with_help() {
        System.out.printf("Usage: train [options] training_set_file [model_file]%n" //
            + "options:%n"
            + "-s type : set type of solver (default 1)%n"
            + "   0 -- L2-regularized logistic regression (primal)%n"
            + "   1 -- L2-regularized L2-loss support vector classification (dual)%n"
            + "   2 -- L2-regularized L2-loss support vector classification (primal)%n"
            + "   3 -- L2-regularized L1-loss support vector classification (dual)%n"
            + "   4 -- multi-class support vector classification by Crammer and Singer%n"
            + "   5 -- L1-regularized L2-loss support vector classification%n"
            + "   6 -- L1-regularized logistic regression%n"
            + "   7 -- L2-regularized logistic regression (dual)%n"
            + "-c cost : set the parameter C (default 1)%n"
            + "-e epsilon : set tolerance of termination criterion%n"
            + "   -s 0 and 2%n"
            + "       |f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,%n"
            + "       where f is the primal function and pos/neg are # of%n"
            + "       positive/negative data (default 0.01)%n"
            + "   -s 1, 3, 4 and 7%n"
            + "       Dual maximal violation <= eps; similar to libsvm (default 0.1)%n"
            + "   -s 5 and 6%n"
            + "       |f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,%n"
            + "       where f is the primal function (default 0.01)%n"
            + "-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)%n"
            + "-wi weight: weights adjust the parameter C of different classes (see README for details)%n"
            + "-v n: n-fold cross validation mode%n"
            + "-q : quiet mode (no outputs)%n");
        System.exit(1);
    }


    Problem getProblem() {
        return prob;
    }

    double getBias() {
        return bias;
    }

    Parameter getParameter() {
        return param;
    }

    void parse_command_line(String argv[]) {
        int i;

        // eps: see setting below
        param = new Parameter(SolverType.L2R_L2LOSS_SVC_DUAL, 1, Double.POSITIVE_INFINITY);
        // default values
        bias = -1;
        cross_validation = false;

        int nr_weight = 0;

        // parse options
        for (i = 0; i < argv.length; i++) {
            if (argv[i].charAt(0) != '-') break;
            if (++i >= argv.length) exit_with_help();
            switch (argv[i - 1].charAt(1)) {
                case 's':
                    param.solverType = SolverType.values()[atoi(argv[i])];
                    break;
                case 'c':
                    param.setC(atof(argv[i]));
                    break;
                case 'e':
                    param.setEps(atof(argv[i]));
                    break;
                case 'B':
                    bias = atof(argv[i]);
                    break;
                case 'w':
                    ++nr_weight;
                    int weightLabel = atoi(argv[i - 1].substring(2));
                    double weight = atof(argv[i]);
                    param.weightLabel = addToArray(param.weightLabel, weightLabel);
                    param.weight = addToArray(param.weight, weight);
                    break;
                case 'v':
                    cross_validation = true;
                    nr_fold = atoi(argv[i]);
                    if (nr_fold < 2) {
                        System.err.println("n-fold cross validation: n must >= 2");
                        exit_with_help();
                    }
                    break;
                case 'q':
                    Linear.disableDebugOutput();
                    break;
                default:
                    System.err.println("unknown option");
                    exit_with_help();
            }
        }

        // determine filenames

        if (i >= argv.length) exit_with_help();

        inputFilename = argv[i];

        if (i < argv.length - 1)
            modelFilename = argv[i + 1];
        else {
            int p = argv[i].lastIndexOf('/');
            ++p; // whew...
            modelFilename = argv[i].substring(p) + ".model";
        }

        if (param.eps == Double.POSITIVE_INFINITY) {
            if (param.solverType == SolverType.L2R_LR || param.solverType == SolverType.L2R_L2LOSS_SVC) {
                param.setEps(0.01);
            } else if (param.solverType == SolverType.L2R_L2LOSS_SVC_DUAL || param.solverType == SolverType.L2R_L1LOSS_SVC_DUAL
                || param.solverType == SolverType.MCSVM_CS || param.solverType == SolverType.L2R_LR_DUAL) {
                param.setEps(0.1);
            } else if (param.solverType == SolverType.L1R_L2LOSS_SVC || param.solverType == SolverType.L1R_LR) {
                param.setEps(0.01);
            }
        }
    }

    /**
     * reads a problem from LibSVM format
     * @param file the SVM file
     * @throws IOException obviously in case of any I/O exception ;)
     * @throws InvalidInputDataException if the input file is not correctly formatted
     */
    public static Problem readProblem(File file, double bias) throws IOException, InvalidInputDataException {
        BufferedReader fp = new BufferedReader(new FileReader(file));
        List<Integer> vy = new ArrayList<Integer>();
        List<Feature[]> vx = new ArrayList<Feature[]>();
        int max_index = 0;

        int lineNr = 0;

        try {
            while (true) {
                String line = fp.readLine();
                if (line == null) break;
                lineNr++;

                StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");
                String token;
                try {
                    token = st.nextToken();
                } catch (NoSuchElementException e) {
                    throw new InvalidInputDataException("empty line", file, lineNr, e);
                }

                try {
                    vy.add(atoi(token));
                } catch (NumberFormatException e) {
                    throw new InvalidInputDataException("invalid label: " + token, file, lineNr, e);
                }

                int m = st.countTokens() / 2;
                Feature[] x;
                if (bias >= 0) {
                    x = new Feature[m + 1];
                } else {
                    x = new Feature[m];
                }
                int indexBefore = 0;
                for (int j = 0; j < m; j++) {

                    token = st.nextToken();
                    int index;
                    try {
                        index = atoi(token);
                    } catch (NumberFormatException e) {
                        throw new InvalidInputDataException("invalid index: " + token, file, lineNr, e);
                    }

                    // assert that indices are valid and sorted
                    if (index < 0) throw new InvalidInputDataException("invalid index: " + index, file, lineNr);
                    if (index <= indexBefore) throw new InvalidInputDataException("indices must be sorted in ascending order", file, lineNr);
                    indexBefore = index;

                    token = st.nextToken();
                    try {
                        double value = atof(token);
                        x[j] = new FeatureNode(index, value);
                    } catch (NumberFormatException e) {
                        throw new InvalidInputDataException("invalid value: " + token, file, lineNr);
                    }
                }
                if (m > 0) {
                    max_index = Math.max(max_index, x[m - 1].getIndex());
                }

                vx.add(x);
            }

            return constructProblem(vy, vx, max_index, bias);
        }
        finally {
            fp.close();
        }
    }

    void readProblem(String filename) throws IOException, InvalidInputDataException {
        prob = Train.readProblem(new File(filename), bias);
    }

    private static int[] addToArray(int[] array, int newElement) {
        int length = array != null ? array.length : 0;
        int[] newArray = new int[length + 1];
        if (array != null && length > 0) {
            System.arraycopy(array, 0, newArray, 0, length);
        }
        newArray[length] = newElement;
        return newArray;
    }

    private static double[] addToArray(double[] array, double newElement) {
        int length = array != null ? array.length : 0;
        double[] newArray = new double[length + 1];
        if (array != null && length > 0) {
            System.arraycopy(array, 0, newArray, 0, length);
        }
        newArray[length] = newElement;
        return newArray;
    }

    private static Problem constructProblem(List<Integer> vy, List<Feature[]> vx, int max_index, double bias) {
        Problem prob = new Problem();
        prob.bias = bias;
        prob.l = vy.size();
        prob.n = max_index;
        if (bias >= 0) {
            prob.n++;
        }
        prob.x = new Feature[prob.l][];
        for (int i = 0; i < prob.l; i++) {
            prob.x[i] = vx.get(i);

            if (bias >= 0) {
                assert prob.x[i][prob.x[i].length - 1] == null;
                prob.x[i][prob.x[i].length - 1] = new FeatureNode(max_index + 1, bias);
            }
        }

        prob.y = new int[prob.l];
        for (int i = 0; i < prob.l; i++)
            prob.y[i] = vy.get(i);

        return prob;
    }

    private void run(String[] args) throws IOException, InvalidInputDataException {
        parse_command_line(args);
        readProblem(inputFilename);
        if (cross_validation)
            do_cross_validation();
        else {
            Model model = Linear.train(prob, param);
            Linear.saveModel(new File(modelFilename), model);
        }
    }
}
