#ifndef SELECTION_METRIC_HXX
#define SELECTION_METRIC_HXX

#include <string>
using std::string;

#include <vector>
using std::vector;

/**
 * Which quantity genome selection optimizes: validation MSE (the default), the EWMA of the
 * cross-sectional rank IC, or the IC gated on MSE. The IC metrics are pooled panel only.
 */
enum SelectionMetric { SELECTION_MSE = 0, SELECTION_IC = 1, SELECTION_IC_GATED = 2 };

/**
 * Process-wide selection settings, parsed once from the command line. Global because
 * RNN_Genome::get_fitness() has no path to per-run configuration. The defaults reproduce
 * the historical MSE-only behavior.
 */
class SelectionConfig {
   private:
    static SelectionMetric metric;
    static bool pooled_panel;
    static int32_t num_stocks;
    static int32_t ic_ewma_halflife;
    static double ic_gate_factor;
    static double max_pred_sd_ratio;

   public:
    /**
     * Parses --selection_metric / --ic_ewma_halflife / --ic_gate_factor.
     *
     * \param _num_stocks the panel width S (1 when not pooled)
     */
    static void initialize_from_arguments(const vector<string>& arguments, bool _pooled_panel, int32_t _num_stocks);

    static SelectionMetric get_metric() {
        return metric;
    }

    /** true when the selection metric itself is IC-based. */
    static bool uses_ic() {
        return metric == SELECTION_IC || metric == SELECTION_IC_GATED;
    }

    /** true when ineligible (high-MSE) genomes must sort last. */
    static bool gates_by_mse() {
        return metric == SELECTION_IC_GATED;
    }

    /** true when a cross-sectional IC can be formed at all; it is logged even under MSE selection. */
    static bool ic_available() {
        return pooled_panel && num_stocks >= 3;
    }

    static bool is_pooled_panel() {
        return pooled_panel;
    }
    static int32_t get_num_stocks() {
        return num_stocks;
    }
    static int32_t get_ic_ewma_halflife() {
        return ic_ewma_halflife;
    }
    static double get_ic_gate_factor() {
        return ic_gate_factor;
    }

    /**
     * Ceiling on (SD of a genome's predictions) / (SD of the targets); a genome above it is
     * treated as unfit. Non-positive disables the guard.
     */
    static double get_max_pred_sd_ratio() {
        return max_pred_sd_ratio;
    }
    static bool guards_prediction_sd() {
        return max_pred_sd_ratio > 0.0;
    }

    /** true when evaluate_online() has to materialize the predictions, not just the MSE. */
    static bool needs_predictions() {
        return ic_available() || guards_prediction_sd();
    }

    /** EWMA smoothing factor derived from the half-life in generations. */
    static double get_ic_ewma_alpha();

    static string get_metric_name();
};

/**
 * Mean daily cross-sectional Spearman rank IC over a pooled-panel validation set. Series are
 * laid out window-major / stock-minor, so v = window * S + stock and each group of S series is
 * one contemporaneous cross-section. Returns NAN if no cross-section contributed.
 *
 * \param num_cross_sections out-param: how many cross-sections contributed
 */
double cross_sectional_rank_ic(
    const vector<vector<vector<double> > >& predictions, const vector<vector<vector<double> > >& expected,
    int32_t num_stocks, int32_t& num_cross_sections
);

/** Spearman rank correlation between two equal-length samples. NAN when either side is constant. */
double spearman_rank_correlation(const vector<double>& a, const vector<double>& b);

/**
 * Ratio of the SD of a genome's predictions to that of the targets, pooled over every
 * (series, output, timestep) value. Returns NAN when the targets have no spread.
 */
double prediction_sd_ratio(
    const vector<vector<vector<double> > >& predictions, const vector<vector<vector<double> > >& expected
);

#endif
