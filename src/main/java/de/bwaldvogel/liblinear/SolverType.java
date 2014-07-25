

package de.bwaldvogel.liblinear;

public enum SolverType {

    /**
     * L2-regularized logistic regression (primal)
     *
     * (fka L2_LR)
     */
    L2R_LR(true),

    /**
     * L2-regularized L2-loss support vector classification (dual)
     *
     * (fka L2LOSS_SVM_DUAL)
     */
    L2R_L2LOSS_SVC_DUAL(false),

    /**
     * L2-regularized L2-loss support vector classification (primal)
     *
     * (fka L2LOSS_SVM)
     */
    L2R_L2LOSS_SVC(false),

    /**
     * L2-regularized L1-loss support vector classification (dual)
     *
     * (fka L1LOSS_SVM_DUAL)
     */
    L2R_L1LOSS_SVC_DUAL(false),

    /**
     * multi-class support vector classification by Crammer and Singer
     */
    MCSVM_CS(false),

    /**
     * L1-regularized L2-loss support vector classification
     *
     * @since 1.5
     */
    L1R_L2LOSS_SVC(false),

    /**
     * L1-regularized logistic regression
     *
     * @since 1.5
     */
    L1R_LR(true),

    /**
     * L2-regularized logistic regression (dual)
     *
     * @since 1.7
     */
    L2R_LR_DUAL(true);

    private final boolean logisticRegressionSolver;

    private SolverType( boolean logisticRegressionSolver ) {
        this.logisticRegressionSolver = logisticRegressionSolver;
    }

    /**
     * @since 1.9
     */
    public boolean isLogisticRegressionSolver() {
        return logisticRegressionSolver;
    }
}
