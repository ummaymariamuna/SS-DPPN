"""
Calibration and Statistical Evaluation Metrics

This module provides comprehensive evaluation metrics including calibration analysis,
statistical tests, and confidence interval estimation for model evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import ( 
    accuracy_score,precision_score,recall_score, f1_score,  
    roc_auc_score,average_precision_score,roc_curve, 
    precision_recall_curve,brier_score_loss
)


try:
    from sklearn.calibration import calibration_curve
    from sklearn.calibration import CalibrationDisplay
    HAS_CALIBRATION = True
except ImportError:
    #Fallback for older sklearn versions
    HAS_CALIBRATION = False
    def calibration_curve(y_true, y_prob, n_bins=10):
        """Fallback implementation of calibration curve"""
        bin_boundaries=np.linspace(0,1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        bin_true = []
        bin_pred = []

        for i in range(n_bins):
            in_bin = (y_prob > bin_lowers[i]) & (y_prob <= bin_uppers[i])
            if i == 0:
                in_bin |= (y_prob == 0)
            # skip empty bins (same as sklearn), don't plot fake points at 0
            if np.sum( in_bin) > 0:
                bin_true.append(np.mean(y_true[in_bin]))
                bin_pred.append(np.mean(y_prob[in_bin]))

        return np.array(bin_true), np.array(bin_pred)

from scipy import stats
from scipy.stats import bootstrap

#Try to import statsmodels for McNemar test
try:
    from statsmodels.stats.contingency_tables import mcnemar 
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False



class CalibrationMetrics:
    """
    Comprehensive calibration and reliability metrics for model evaluation
    
    This class provides methods for evaluating model calibration including:
    - Expected Calibration Error (ECE)
    - Reliability diagrams
    - Brier score decomposition
    - Bootstrap confidence intervals
    """

    @staticmethod
    def expected_calibration_error(y_true,y_prob,n_bins=10):
        """
        Calculate Expected Calibration Error (ECE)
        
        Args:
            y_true (array-like): True binary labels
            y_prob (array-like): Predicted probabilities for positive class
            n_bins (int): Number of bins for calibration
        
        Returns:
            float: Expected Calibration Error
        """
        bin_boundaries=np.linspace(0,1,n_bins+1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        ece = 0
        total_samples = len(y_true)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin (first bin also takes prob == 0)
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            if bin_lower == 0:
                in_bin |= (y_prob == 0)
            prop_in_bin = in_bin.sum() / total_samples

            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean( )
                
                ece += np.abs( avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin



        
        return ece
    
    @staticmethod
    def brier_score(y_true, y_prob):
        """
        Calculate Brier score
        
        Args:
            y_true (array-like): True binary labels (0 or 1)
            y_prob (array-like): Predicted probabilities for positive class
        
        Returns:
            float: Brier score (lower is better)
        """
        return brier_score_loss( y_true, y_prob)
    
    @staticmethod
    def reliability_curve(y_true, y_prob, n_bins=10):
        """
        Calculate reliability curve data
        
        Args:
            y_true (array-like): True binary labels
            y_prob (array-like): Predicted probabilities
            n_bins (int): Number of bins
        
        Returns:
            tuple: (bin_centers, accuracies, confidences, counts)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1) 
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        bin_centers=[]
        accuracies = []
        confidences = []
        counts = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            if bin_lower == 0:
                in_bin |= (y_prob == 0)
            count = in_bin.sum()

            
            if count > 0:
                accuracy = y_true[in_bin].mean() 
                confidence = y_prob[in_bin].mean()
                
                bin_centers.append((bin_lower + bin_upper) / 2)
                accuracies.append(accuracy)
                confidences.append(confidence)
                counts.append(count)

        return np.array(bin_centers), np.array(accuracies), np.array(confidences), np.array(counts)
    
    @staticmethod
    def plot_calibration_curve(y_true, y_prob, model_name="Model", n_bins=10, save_path=None):
        """
        Plot calibration curve with reliability diagram
        
        Args:
            y_true (array-like): True labels
            y_prob (array-like): Predicted probabilities
            model_name (str): Model name for plot title
            n_bins (int): Number of bins
            save_path (str): Path to save the plot
        
        Returns:
            tuple: (figure, (ax1, ax2)) matplotlib objects
        """
        fig, (ax1, ax2) = plt.subplots(  1, 2, figsize=(12, 5))

        fraction_of_positives, mean_predicted_value=calibration_curve(
            y_true, y_prob, n_bins=n_bins
        )
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax1.plot(mean_predicted_value, fraction_of_positives, 'bo-', 
                label=f'{model_name}', markersize=6, linewidth=2)
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title(f'Calibration Curve - {model_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        bin_centers, accuracies, confidences, counts = CalibrationMetrics.reliability_curve(
            y_true, y_prob, n_bins
        )
        
        ax2.bar(bin_centers, counts, width = 0.08, alpha=0.7, 
               label='Sample count', color='lightblue')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(bin_centers, accuracies, 'ro-', label='Accuracy', markersize=6)
        ax2_twin.plot(bin_centers, confidences, 'bo-', label='Confidence', markersize=6)
        ax2_twin.plot([0, 1], [0, 1], 'k--', alpha=0.5)

        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Count')
        ax2_twin.set_ylabel('Accuracy/Confidence')
        ax2.set_title(f'Reliability Diagram - {model_name}')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3) 

        plt.tight_layout()



        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig,(ax1, ax2)


class StatisticalTests:
    """
    Statistical significance tests for model comparison
    
    Provides methods for comparing model performance using various statistical tests
    including DeLong test for AUC comparison and McNemar test for accuracy comparison.
    """

    @staticmethod
    def delong_test( y_true, y_prob1, y_prob2): 
        """
        DeLong test for comparing two AUC scores
        
        Args:
            y_true (array-like): True labels
            y_prob1 (array-like): Predicted probabilities from model 1
            y_prob2 (array-like): Predicted probabilities from model 2
        
        Returns:
            tuple: (z_score, p_value)
        """
        y_true = np.asarray(y_true)
        y_prob1 = np.asarray(y_prob1)
        y_prob2=np.asarray(y_prob2)

        auc1 = roc_auc_score(y_true, y_prob1)
        auc2 = roc_auc_score(y_true, y_prob2)

        n1 = np.sum(y_true == 1)
        n0 = np.sum(y_true == 0)

        # psi = 1 if pos > neg, 0.5 for ties, 0 otherwise
        def components(prob):
            pos_scores = prob[y_true == 1]
            neg_scores = prob[y_true == 0]
            psi = (pos_scores[:,None] > neg_scores[None,:]) + 0.5 * (pos_scores[:,None] == neg_scores[None, :])
            return psi.mean(axis=1), psi.mean(axis=0)  #V10 (per positive), V01 (per negative)

        v10_1, v01_1 = components(y_prob1)
        v10_2, v01_2=components(y_prob2)

        # covariance between the two models, since they're scored on the same samples
        s10=np.cov( np.vstack([v10_1, v10_2]))
        s01 = np.cov(np.vstack([v01_1, v01_2]))
        var_diff = ((s10[0,0] + s10[1,1] - 2 * s10[0,1]) / n1 +
                    (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n0)
        if var_diff <= 0:
            return 0,1.0
        se_diff = np.sqrt(var_diff)
        z_score=(auc1 - auc2) / se_diff
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_score)))

        return z_score,p_value
    @staticmethod
    def mcnemar_test(y_true, y_pred1, y_pred2): 
        """
        McNemar test for comparing two classifiers
        
        Args:
            y_true (array-like): True labels
            y_pred1 (array-like): Predictions from model 1
            y_pred2 (array-like): Predictions from model 2
        
        Returns:
            tuple: (statistic, p_value)
        """
        if HAS_STATSMODELS:
            correct1 = (y_pred1 == y_true)
            correct2 = (y_pred2 == y_true)
            
            # McNemar table: [both_correct, model1_correct_model2_wrong, 
            #model1_wrong_model2_correct, both_wrong]
            both_correct = np.sum(correct1 & correct2)
            model1_only = np.sum(correct1 & ~correct2)
            model2_only = np.sum(~correct1 & correct2)
            both_wrong = np.sum(~correct1 & ~correct2)
            
            table=np.array([[both_correct, model1_only],
                             [model2_only, both_wrong]])
            result = mcnemar(table, exact = False, correction=True)
            return result.statistic, result.pvalue
        else:
            correct1 = (y_pred1 == y_true)
            correct2 = (y_pred2 == y_true)
            model1_only = np.sum(correct1 & ~correct2)
            model2_only = np.sum(~correct1 & correct2)
            
            if model1_only + model2_only == 0:
                return 0, 1.0

            statistic = (abs(model1_only - model2_only) - 1)**2 / (model1_only + model2_only)
            p_value = 1 - stats.chi2.cdf(statistic,df=1)
            
            return statistic, p_value

    @staticmethod
    def bootstrap_metric(y_true,y_prob,y_pred,metric_func, n_bootstrap = 1000, random_state=42):
        """
        Bootstrap confidence intervals for a metric
        
        Args:
            y_true (array-like): True labels
            y_prob (array-like): Predicted probabilities (can be None for some metrics)
            y_pred (array-like): Predicted labels (can be None for some metrics)
            metric_func (callable): Metric function to bootstrap
            n_bootstrap (int): Number of bootstrap samples
            random_state (int): Random seed
        
        Returns:
            dict: Dictionary with 'mean', 'std', 'ci_lower', 'ci_upper'
        """
        np.random.seed(random_state)
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob) if y_prob is not None else None
        y_pred = np.asarray(y_pred) if y_pred is not None else None
        n_samples = len(y_true)
        bootstrap_scores = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            
            y_true_boot = y_true[indices]
            y_prob_boot=y_prob[indices] if y_prob is not None else None
            y_pred_boot = y_pred[indices] if y_pred is not None else None
            
            # Calculate metric
            try: 
                if y_prob_boot is not None and y_pred_boot is not None:
                    score = metric_func(y_true_boot, y_prob_boot, y_pred_boot)
                elif y_prob_boot is not None:
                    score = metric_func( y_true_boot, y_prob_boot)
                elif y_pred_boot is not None:
                    score = metric_func(y_true_boot, y_pred_boot)
                else:
                    score = metric_func(y_true_boot)

                bootstrap_scores.append(score)
            except ValueError:
                continue  # Skip if metric calculation fails (e.g. one class in the resample)

        bootstrap_scores = np.array(bootstrap_scores)

        if len(bootstrap_scores) == 0:
            print(f"Warning: all {n_bootstrap} bootstrap samples failed")
            return {'mean': np.nan, 'std': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}

        return {
            'mean': np.mean(bootstrap_scores),
            'std': np.std(bootstrap_scores),
            'ci_lower': np.percentile(bootstrap_scores, 2.5),
            'ci_upper': np.percentile(bootstrap_scores, 97.5)
        }


class ComprehensiveEvaluator: 
    """
    Comprehensive model evaluation with calibration and statistical analysis
    
    This class provides a complete evaluation pipeline including:
    - Standard classification metrics
    - Calibration analysis
    - Bootstrap confidence intervals
    - Statistical significance tests
    """
    
    def __init__( self, n_bootstrap=1000, random_state=42):
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.calibration = CalibrationMetrics()
        self.stats = StatisticalTests()

    def evaluate_with_confidence_intervals(self, y_true, y_prob, model_name="Model", 
                                         save_prefix="evaluation"):
        """
        Comprehensive evaluation with confidence intervals
        
        Args:
            y_true (array-like): True labels
            y_prob (array-like): Predicted probabilities
            model_name (str): Model name for reporting
            save_prefix (str): Prefix for saved files
        
        Returns:
            dict: Comprehensive evaluation results
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        #Convert probabilities to predictions
        y_pred=(y_prob > 0.5).astype(int)

        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE EVALUATION: {model_name}")
        print(f"{'='*60}")
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        try:
            auroc = roc_auc_score(y_true, y_prob)
        except:
            auroc = 0.5
        
        try:
            auprc = average_precision_score(y_true, y_prob)
        except:
            auprc = np.mean(y_true)  # Baseline performance

        brier_score = brier_score_loss( y_true, y_prob)
        ece=self.calibration.expected_calibration_error(y_true,y_prob)

        # Bootstrap confidence intervals
        print("Calculating bootstrap confidence intervals...")
        
        metrics_to_bootstrap = {
            'accuracy': lambda yt, yp: accuracy_score(yt, (yp > 0.5).astype(int)),
            'precision': lambda yt, yp: precision_score(yt, (yp > 0.5).astype(int), 
                                                       average='binary', zero_division=0),
            'recall': lambda yt, yp: recall_score(yt, (yp > 0.5).astype(int), 
                                                 average='binary', zero_division=0),
            'f1': lambda yt, yp: f1_score(yt, (yp > 0.5).astype(int), 
                                         average='binary', zero_division=0),
            'auroc': lambda yt, yp: roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else 0.5,
            'auprc': lambda yt, yp: average_precision_score(yt, yp) if len(np.unique(yt)) > 1 else np.mean(yt),
            'brier_score': lambda yt, yp: brier_score_loss(yt, yp),
            'ece': lambda yt, yp: self.calibration.expected_calibration_error(yt, yp)
        }

        
        # otherwise bootstrap_metric calls them with 3 args and every sample fails
        bootstrap_results = {}
        for metric_name, metric_func in metrics_to_bootstrap.items():
            bootstrap_results[metric_name] = self.stats.bootstrap_metric(
                y_true, y_prob, None, metric_func,
                self.n_bootstrap, self.random_state
            )
        
        results = { 
            'model_name': model_name,
            'n_samples': len(y_true),
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auroc': auroc,
                'auprc': auprc,
                'brier_score': brier_score,
                'ece': ece
            },
            'bootstrap_ci': bootstrap_results,
            'y_true': y_true,
            'y_prob': y_prob,
            'y_pred': y_pred
        }
        
        # Print results
        self._print_results(results)

        self._create_evaluation_plots(results, save_prefix)
        return results

    def compare_models(self, results1, results2, model1_name=None, model2_name=None):
        """
        Statistical comparison between two models
        
        Args:
            results1 (dict): Results from first model
            results2 (dict): Results from second model
            model1_name (str): Name of first model
            model2_name (str): Name of second model
        
        Returns:
            dict: Comparison results
        """
        model1_name = model1_name or results1['model_name']
        model2_name = model2_name or results2['model_name']

        print(f"\n{'='*60}")
        print(f"STATISTICAL COMPARISON: {model1_name} vs {model2_name}")
        print(f"{'='*60}")
        
        y_true = results1['y_true']
        y_prob1 = results1['y_prob']
        y_prob2 = results2['y_prob']
        y_pred1 = results1['y_pred']
        y_pred2 = results2['y_pred']

        delong_stat, delong_p = self.stats.delong_test(y_true, y_prob1, y_prob2)

        mcnemar_stat, mcnemar_p=self.stats.mcnemar_test(y_true, y_pred1, y_pred2)
        
        # Performance differences
        auroc_diff = results1['metrics']['auroc'] - results2['metrics']['auroc']
        accuracy_diff = results1['metrics']['accuracy'] - results2['metrics']['accuracy']
        
        comparison_results = {
            'model1_name': model1_name,
            'model2_name': model2_name,
            'auroc_difference': auroc_diff,
            'accuracy_difference': accuracy_diff,
            'delong_test': {'statistic': delong_stat, 'p_value': delong_p},
            'mcnemar_test': {'statistic': mcnemar_stat, 'p_value': mcnemar_p}
        }

        print(f"AUROC Difference: {auroc_diff:.4f}")
        print(f"Accuracy Difference: {accuracy_diff:.4f}")
        print(f"\nDeLong Test (AUC comparison):")
        print(f"  Statistic: {delong_stat:.4f}")
        print(f"  P-value: {delong_p:.4f}")
        print(f"  Significant: {'Yes' if delong_p < 0.05 else 'No'}")
        print(f"\nMcNemar Test (Accuracy comparison):")
        print(f"  Statistic: {mcnemar_stat:.4f}")
        print(f"  P-value: {mcnemar_p:.4f}")
        print(f"  Significant: {'Yes' if mcnemar_p < 0.05 else 'No'}")
        
        return comparison_results

    def _print_results(self, results):
        """Print evaluation results in a formatted way"""
        print(f"\nModel: {results['model_name']}")
        print(f"Samples: {results['n_samples']}")
        print(f"\n{'Metric':<15} {'Value':<10} {'95% CI':<20}")
        print("-" * 50)
        
        for metric_name, value in results['metrics'].items():
            ci = results['bootstrap_ci'][metric_name]
            ci_str = f"[{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]"
            print(f"{metric_name:<15} {value:<10.3f} {ci_str:<20}")
    def _create_evaluation_plots( self, results, save_prefix):
        """Create comprehensive evaluation plots"""
        y_true = results['y_true']
        y_prob = results['y_prob']
        model_name = results['model_name']
        
        fig = plt.figure(figsize=(15,10))
        
        # ROC Curve
        ax1 = plt.subplot(2,3,1) 
        fpr, tpr, _ = roc_curve(y_true, y_prob) 
        auroc = results['metrics']['auroc']
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auroc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True,alpha=0.3)


        ax2 = plt.subplot(2, 3, 2)
        precision,recall,_ = precision_recall_curve(y_true,y_prob)
        auprc = results['metrics']['auprc']
        plt.plot(recall, precision, linewidth=2, label=f'{model_name} (AUC = {auprc:.3f})')
        plt.axhline(y=np.mean(y_true), color='k', linestyle='--', alpha=0.5, 
                   label='Baseline')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        ax3 = plt.subplot(2, 3, 3)
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins = 10
        )
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        plt.plot(mean_predicted_value, fraction_of_positives, 'bo-', 
                label=f'{model_name}', markersize=6, linewidth=2)
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend(  )
        plt.grid(True,alpha=0.3)
        
        ax4 = plt.subplot(2, 3, 4)
        bin_centers, accuracies, confidences, counts = self.calibration.reliability_curve(
            y_true, y_prob, n_bins=10
        )
        plt.bar( bin_centers,counts, width = 0.08, alpha=0.7, 
               label='Sample count', color='lightblue')
        ax4_twin=ax4.twinx()
        ax4_twin.plot(bin_centers, accuracies, 'ro-', label='Accuracy', markersize=6)
        ax4_twin.plot(bin_centers, confidences, 'bo-', label='Confidence', markersize=6)
        ax4_twin.plot([0, 1], [0, 1], 'k--', alpha=0.5)

        ax4.set_xlabel('Confidence')
        ax4.set_ylabel('Count')
        ax4_twin.set_ylabel('Accuracy/Confidence')
        ax4.set_title('Reliability Diagram')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        ax4.grid(True,alpha=0.3)

        #Prediction Histogram
        ax5 = plt.subplot(2,3,5) 
        plt.hist(y_prob[y_true == 0], bins=20, alpha=0.7, label='Negative', density=True)
        plt.hist(y_prob[y_true == 1], bins=20, alpha=0.7, label='Positive', density=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Prediction Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        #Metrics Bar Plot
        ax6=plt.subplot(2, 3, 6)
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUROC', 'AUPRC']
        metrics_values = [
            results['metrics']['accuracy'],
            results['metrics']['precision'],
            results['metrics']['recall'],
            results['metrics']['f1_score'],
            results['metrics']['auroc'],
            results['metrics']['auprc']
        ]
        bars = plt.bar(metrics_names,metrics_values, alpha = 0.7)
        plt.ylabel('Score')
        plt.title('Performance Metrics')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)


        for bar,value in zip(bars,metrics_values):
            plt.text(bar.get_x() + bar.get_width()/2,bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        # Save the plot
        save_path = f"{save_prefix}_comprehensive_evaluation.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evaluation plots saved to: {save_path}")
        plt.show()
        
        return fig


# Aliases for backward compatibility and cleaner naming
CalibrationEvaluator = CalibrationMetrics

#Add method aliases
CalibrationMetrics.plot_reliability_diagram = CalibrationMetrics.plot_calibration_curve
