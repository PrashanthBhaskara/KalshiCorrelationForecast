"""
Forecast Comparison: Kalshi vs CME FedWatch for Jan 2026 Fed Decision

Compares the forecasting accuracy of:
1. Weighted combination of two Kalshi markets (kxfeddecision + kxfedcombo)
2. CME FedWatch

Actual outcome: 25 bps cut with dissent > 0 (Jan 28, 2026)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class ForecastComparison:
    def __init__(self, cme_path: str, kalshi_decision_path: str, kalshi_combo_path: str):
        """Initialize with paths to the three data sources."""
        self.cme_path = cme_path
        self.kalshi_decision_path = kalshi_decision_path
        self.kalshi_combo_path = kalshi_combo_path

        # Actual outcome: 25 bps cut with dissent > 0
        self.actual_outcome = 1  # Binary: 1 = 25 bps cut happened, 0 = didn't happen

        self.cme_df = None
        self.kalshi_decision_df = None
        self.kalshi_combo_df = None
        self.aligned_df = None

    def load_data(self):
        """Load and preprocess all datasets."""
        print("="*80)
        print("LOADING DATA")
        print("="*80)

        # Load CME FedWatch
        print("\n1. CME FedWatch data...")
        self.cme_df = pd.read_csv(self.cme_path)
        self.cme_df.columns = [col.strip() for col in self.cme_df.columns]
        self.cme_df = self.cme_df.rename(columns={'Date': 'date'})
        self.cme_df['date'] = pd.to_datetime(self.cme_df['date'])

        # CME probabilities are already in 0-1 range
        # Column (325-350) represents 25 bps cut outcome
        # (Rate was at 350-375 before meeting, cut to 325-350)
        self.cme_df['cme_prob_25cut'] = self.cme_df['(325-350)']

        print(f"   Loaded {len(self.cme_df)} observations")
        print(f"   Date range: {self.cme_df['date'].min()} to {self.cme_df['date'].max()}")

        # Load Kalshi decision market
        print("\n2. Kalshi decision market...")
        self.kalshi_decision_df = pd.read_csv(self.kalshi_decision_path)
        self.kalshi_decision_df['timestamp'] = pd.to_datetime(self.kalshi_decision_df['timestamp']).dt.tz_localize(None)

        # Convert to probabilities if needed
        decision_cols = [c for c in self.kalshi_decision_df.columns if c != 'timestamp']
        for col in decision_cols:
            if self.kalshi_decision_df[col].max() > 1.5:
                self.kalshi_decision_df[col] = self.kalshi_decision_df[col] / 100.0

        # Extract 25 bps cut probability
        if 'Cut 25bps' in self.kalshi_decision_df.columns:
            self.kalshi_decision_df['kalshi_decision_prob_25cut'] = self.kalshi_decision_df['Cut 25bps']

        print(f"   Loaded {len(self.kalshi_decision_df)} observations")
        print(f"   Date range: {self.kalshi_decision_df['timestamp'].min()} to {self.kalshi_decision_df['timestamp'].max()}")

        # Load Kalshi combo market
        print("\n3. Kalshi combo market...")
        self.kalshi_combo_df = pd.read_csv(self.kalshi_combo_path)
        self.kalshi_combo_df['timestamp'] = pd.to_datetime(self.kalshi_combo_df['timestamp']).dt.tz_localize(None)

        # Convert to probabilities if needed
        combo_cols = [c for c in self.kalshi_combo_df.columns if c != 'timestamp']
        for col in combo_cols:
            if self.kalshi_combo_df[col].max() > 1.5:
                self.kalshi_combo_df[col] = self.kalshi_combo_df[col] / 100.0

        # Extract 25 bps cut probability (sum of both dissent statuses)
        cut_25_cols = [c for c in self.kalshi_combo_df.columns if '25bp cut' in c]
        if cut_25_cols:
            self.kalshi_combo_df['kalshi_combo_prob_25cut'] = self.kalshi_combo_df[cut_25_cols].sum(axis=1)
            print(f"   Found columns for 25bp cut: {cut_25_cols}")

        print(f"   Loaded {len(self.kalshi_combo_df)} observations")
        print(f"   Date range: {self.kalshi_combo_df['timestamp'].min()} to {self.kalshi_combo_df['timestamp'].max()}")

    def align_data(self):
        """Align all datasets by timestamp."""
        print("\n" + "="*80)
        print("ALIGNING DATA")
        print("="*80)

        # Rename CME date column to timestamp for consistency
        self.cme_df = self.cme_df.rename(columns={'date': 'timestamp'})

        # Merge Kalshi markets
        kalshi_merged = pd.merge(
            self.kalshi_decision_df[['timestamp', 'kalshi_decision_prob_25cut']],
            self.kalshi_combo_df[['timestamp', 'kalshi_combo_prob_25cut']],
            on='timestamp',
            how='outer'
        ).sort_values('timestamp')

        # Merge with CME
        self.aligned_df = pd.merge(
            kalshi_merged,
            self.cme_df[['timestamp', 'cme_prob_25cut']],
            on='timestamp',
            how='outer'
        ).sort_values('timestamp')

        # Forward fill missing values
        self.aligned_df = self.aligned_df.ffill()

        # Drop rows with any remaining NaN
        self.aligned_df = self.aligned_df.dropna()

        print(f"Aligned dataset: {len(self.aligned_df)} observations")
        print(f"Date range: {self.aligned_df['timestamp'].min()} to {self.aligned_df['timestamp'].max()}")

        return self.aligned_df

    def create_weighted_kalshi_forecast(self, weight_decision=0.5, weight_combo=0.5):
        """
        Create weighted combination of two Kalshi markets.

        Args:
            weight_decision: Weight for kxfeddecision market (default 0.5)
            weight_combo: Weight for kxfedcombo market (default 0.5)
        """
        print("\n" + "="*80)
        print("CREATING WEIGHTED KALSHI FORECAST")
        print("="*80)

        # Normalize weights
        total_weight = weight_decision + weight_combo
        w1 = weight_decision / total_weight
        w2 = weight_combo / total_weight

        self.aligned_df['kalshi_weighted_prob_25cut'] = (
            w1 * self.aligned_df['kalshi_decision_prob_25cut'] +
            w2 * self.aligned_df['kalshi_combo_prob_25cut']
        )

        print(f"Weights: Decision={w1:.2f}, Combo={w2:.2f}")
        print(f"\nWeighted Kalshi forecast statistics:")
        print(f"  Mean: {self.aligned_df['kalshi_weighted_prob_25cut'].mean():.4f}")
        print(f"  Median: {self.aligned_df['kalshi_weighted_prob_25cut'].median():.4f}")
        print(f"  Std: {self.aligned_df['kalshi_weighted_prob_25cut'].std():.4f}")
        print(f"  Range: [{self.aligned_df['kalshi_weighted_prob_25cut'].min():.4f}, {self.aligned_df['kalshi_weighted_prob_25cut'].max():.4f}]")

        # Also print individual market stats
        print(f"\nKalshi Decision market statistics:")
        print(f"  Mean: {self.aligned_df['kalshi_decision_prob_25cut'].mean():.4f}")
        print(f"  Final forecast: {self.aligned_df['kalshi_decision_prob_25cut'].iloc[-1]:.4f}")

        print(f"\nKalshi Combo market statistics:")
        print(f"  Mean: {self.aligned_df['kalshi_combo_prob_25cut'].mean():.4f}")
        print(f"  Final forecast: {self.aligned_df['kalshi_combo_prob_25cut'].iloc[-1]:.4f}")

        return self.aligned_df['kalshi_weighted_prob_25cut']

    def compute_brier_score(self, probabilities, actual_outcome):
        """
        Compute Brier score for a sequence of probability forecasts.

        Brier Score = mean((forecast - actual)^2)
        Lower is better (perfect score = 0)
        """
        return np.mean((probabilities - actual_outcome) ** 2)

    def compute_log_score(self, probabilities, actual_outcome):
        """
        Compute logarithmic score (log loss).

        Log Score = -log(p) if outcome = 1, -log(1-p) if outcome = 0
        Lower is better
        """
        # Clip probabilities to avoid log(0)
        probs = np.clip(probabilities, 1e-10, 1 - 1e-10)

        if actual_outcome == 1:
            return -np.mean(np.log(probs))
        else:
            return -np.mean(np.log(1 - probs))

    def compute_calibration(self, probabilities, actual_outcome, n_bins=10):
        """
        Compute calibration statistics.

        Returns fraction of time in each probability bin and actual outcome frequency.
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probabilities, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        calibration_data = []
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                mean_prob = probabilities[mask].mean()
                count = mask.sum()
                calibration_data.append({
                    'bin': i,
                    'mean_forecast': mean_prob,
                    'count': count,
                    'bin_lower': bins[i],
                    'bin_upper': bins[i+1]
                })

        return pd.DataFrame(calibration_data)

    def compare_forecasts(self):
        """Compare Kalshi weighted vs CME FedWatch forecasts."""
        print("\n" + "="*80)
        print("FORECAST COMPARISON METRICS")
        print("="*80)
        print(f"\nActual outcome: 25 bps cut (binary = {self.actual_outcome})")

        # Extract final forecasts
        final_kalshi = self.aligned_df['kalshi_weighted_prob_25cut'].iloc[-1]
        final_cme = self.aligned_df['cme_prob_25cut'].iloc[-1]

        print(f"\nFinal forecasts (on meeting day):")
        print(f"  Kalshi weighted: {final_kalshi:.4f} ({final_kalshi*100:.2f}%)")
        print(f"  CME FedWatch:    {final_cme:.4f} ({final_cme*100:.2f}%)")

        # Compute Brier scores
        brier_kalshi = self.compute_brier_score(
            self.aligned_df['kalshi_weighted_prob_25cut'],
            self.actual_outcome
        )
        brier_cme = self.compute_brier_score(
            self.aligned_df['cme_prob_25cut'],
            self.actual_outcome
        )

        print(f"\n" + "-"*80)
        print("BRIER SCORES (lower is better, range 0-1)")
        print("-"*80)
        print(f"  Kalshi weighted: {brier_kalshi:.6f}")
        print(f"  CME FedWatch:    {brier_cme:.6f}")
        print(f"  Difference:      {brier_kalshi - brier_cme:.6f} {'(Kalshi worse)' if brier_kalshi > brier_cme else '(Kalshi better)'}")
        print(f"  Relative improvement: {abs(brier_kalshi - brier_cme) / max(brier_kalshi, brier_cme) * 100:.2f}%")

        # Compute log scores
        log_kalshi = self.compute_log_score(
            self.aligned_df['kalshi_weighted_prob_25cut'],
            self.actual_outcome
        )
        log_cme = self.compute_log_score(
            self.aligned_df['cme_prob_25cut'],
            self.actual_outcome
        )

        print(f"\n" + "-"*80)
        print("LOGARITHMIC SCORES (lower is better)")
        print("-"*80)
        print(f"  Kalshi weighted: {log_kalshi:.6f}")
        print(f"  CME FedWatch:    {log_cme:.6f}")
        print(f"  Difference:      {log_kalshi - log_cme:.6f} {'(Kalshi worse)' if log_kalshi > log_cme else '(Kalshi better)'}")

        # Compute mean absolute error
        mae_kalshi = np.mean(np.abs(self.aligned_df['kalshi_weighted_prob_25cut'] - self.actual_outcome))
        mae_cme = np.mean(np.abs(self.aligned_df['cme_prob_25cut'] - self.actual_outcome))

        print(f"\n" + "-"*80)
        print("MEAN ABSOLUTE ERROR (lower is better)")
        print("-"*80)
        print(f"  Kalshi weighted: {mae_kalshi:.6f}")
        print(f"  CME FedWatch:    {mae_cme:.6f}")
        print(f"  Difference:      {mae_kalshi - mae_cme:.6f} {'(Kalshi worse)' if mae_kalshi > mae_cme else '(Kalshi better)'}")

        # Sharpness (standard deviation of forecasts)
        sharpness_kalshi = self.aligned_df['kalshi_weighted_prob_25cut'].std()
        sharpness_cme = self.aligned_df['cme_prob_25cut'].std()

        print(f"\n" + "-"*80)
        print("SHARPNESS (std dev of forecasts, higher = more dynamic)")
        print("-"*80)
        print(f"  Kalshi weighted: {sharpness_kalshi:.6f}")
        print(f"  CME FedWatch:    {sharpness_cme:.6f}")
        print(f"  Ratio:           {sharpness_kalshi / sharpness_cme:.2f}x")

        # Store results
        self.comparison_results = {
            'Metric': ['Brier Score', 'Log Score', 'MAE', 'Sharpness', 'Final Forecast'],
            'Kalshi Weighted': [brier_kalshi, log_kalshi, mae_kalshi, sharpness_kalshi, final_kalshi],
            'CME FedWatch': [brier_cme, log_cme, mae_cme, sharpness_cme, final_cme],
            'Winner': [
                'CME' if brier_cme < brier_kalshi else 'Kalshi',
                'CME' if log_cme < log_kalshi else 'Kalshi',
                'CME' if mae_cme < mae_kalshi else 'Kalshi',
                'Kalshi' if sharpness_kalshi > sharpness_cme else 'CME',
                'N/A'
            ]
        }

        self.comparison_df = pd.DataFrame(self.comparison_results)

        return self.comparison_df

    def visualize_forecasts(self, save_path='Fed_Analysis/plots'):
        """Create visualizations comparing forecasts."""
        Path(save_path).mkdir(parents=True, exist_ok=True)

        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)

        # Plot 1: Forecast evolution over time
        fig, ax = plt.subplots(figsize=(16, 8))

        ax.plot(self.aligned_df['timestamp'],
                self.aligned_df['kalshi_weighted_prob_25cut'],
                label='Kalshi Weighted', linewidth=2.5, color='blue', alpha=0.8)

        ax.plot(self.aligned_df['timestamp'],
                self.aligned_df['cme_prob_25cut'],
                label='CME FedWatch', linewidth=2.5, color='red', alpha=0.8)

        # Add actual outcome line
        ax.axhline(y=self.actual_outcome, color='green', linestyle='--',
                   linewidth=2, alpha=0.7, label='Actual Outcome (25 bps cut)')

        # Add meeting date vertical line
        meeting_date = pd.to_datetime('2026-01-28')
        ax.axvline(x=meeting_date, color='black', linestyle=':',
                   linewidth=1.5, alpha=0.5, label='Meeting Date')

        ax.set_xlabel('Date', fontsize=14, fontweight='bold')
        ax.set_ylabel('Probability of 25 bps Cut', fontsize=14, fontweight='bold')
        ax.set_title('Forecast Comparison: Kalshi vs CME FedWatch\nJan 2026 Fed Meeting (25 bps cut)',
                     fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_path}/forecast_comparison_timeseries.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}/forecast_comparison_timeseries.png")
        plt.close()

        # Plot 2: Forecast error over time
        fig, ax = plt.subplots(figsize=(16, 8))

        kalshi_error = self.aligned_df['kalshi_weighted_prob_25cut'] - self.actual_outcome
        cme_error = self.aligned_df['cme_prob_25cut'] - self.actual_outcome

        ax.plot(self.aligned_df['timestamp'], kalshi_error,
                label='Kalshi Error', linewidth=2, color='blue', alpha=0.7)

        ax.plot(self.aligned_df['timestamp'], cme_error,
                label='CME Error', linewidth=2, color='red', alpha=0.7)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=meeting_date, color='black', linestyle=':',
                   linewidth=1.5, alpha=0.5, label='Meeting Date')

        ax.set_xlabel('Date', fontsize=14, fontweight='bold')
        ax.set_ylabel('Forecast Error (Forecast - Actual)', fontsize=14, fontweight='bold')
        ax.set_title('Forecast Errors Over Time\n(Negative = Underestimated probability of cut)',
                     fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_path}/forecast_errors.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}/forecast_errors.png")
        plt.close()

        # Plot 3: All three Kalshi forecasts comparison
        fig, ax = plt.subplots(figsize=(16, 8))

        ax.plot(self.aligned_df['timestamp'],
                self.aligned_df['kalshi_decision_prob_25cut'],
                label='Kalshi Decision Market', linewidth=2, color='blue', alpha=0.7)

        ax.plot(self.aligned_df['timestamp'],
                self.aligned_df['kalshi_combo_prob_25cut'],
                label='Kalshi Combo Market', linewidth=2, color='purple', alpha=0.7)

        ax.plot(self.aligned_df['timestamp'],
                self.aligned_df['kalshi_weighted_prob_25cut'],
                label='Kalshi Weighted (50/50)', linewidth=2.5, color='darkblue', alpha=0.9)

        ax.axhline(y=self.actual_outcome, color='green', linestyle='--',
                   linewidth=2, alpha=0.7, label='Actual Outcome')

        ax.axvline(x=meeting_date, color='black', linestyle=':',
                   linewidth=1.5, alpha=0.5, label='Meeting Date')

        ax.set_xlabel('Date', fontsize=14, fontweight='bold')
        ax.set_ylabel('Probability of 25 bps Cut', fontsize=14, fontweight='bold')
        ax.set_title('Kalshi Markets Comparison', fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_path}/kalshi_markets_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}/kalshi_markets_comparison.png")
        plt.close()

        # Plot 4: Metrics comparison bar chart
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics_to_plot = ['Brier Score', 'Log Score', 'MAE', 'Sharpness']

        for idx, (ax, metric) in enumerate(zip(axes.flat, metrics_to_plot)):
            metric_row = self.comparison_df[self.comparison_df['Metric'] == metric].iloc[0]

            values = [metric_row['Kalshi Weighted'], metric_row['CME FedWatch']]
            colors = ['blue', 'red']

            bars = ax.bar(['Kalshi Weighted', 'CME FedWatch'], values, color=colors, alpha=0.7)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

            ax.set_ylabel(metric, fontsize=12, fontweight='bold')
            ax.set_title(f'{metric} Comparison\n({"lower is better" if metric != "Sharpness" else "higher is more dynamic"})',
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f'{save_path}/metrics_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}/metrics_comparison.png")
        plt.close()

    def save_results(self, output_path='Fed_Analysis/forecast_comparison_results.csv'):
        """Save comparison results."""
        self.aligned_df.to_csv(output_path, index=False)
        print(f"\n✅ Saved full forecast data to: {output_path}")

        summary_path = output_path.replace('.csv', '_summary.csv')
        self.comparison_df.to_csv(summary_path, index=False)
        print(f"✅ Saved comparison summary to: {summary_path}")

    def run_full_analysis(self, weight_decision=0.5, weight_combo=0.5):
        """Run complete forecast comparison analysis."""
        print("\n" + "="*80)
        print("FORECAST COMPARISON: KALSHI vs CME FEDWATCH")
        print("January 2026 Fed Meeting Analysis")
        print("="*80)

        self.load_data()
        self.align_data()
        self.create_weighted_kalshi_forecast(weight_decision, weight_combo)
        self.compare_forecasts()
        self.visualize_forecasts()
        self.save_results()

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)

        print("\nSUMMARY:")
        print(self.comparison_df.to_string(index=False))

        return self.aligned_df, self.comparison_df


def main():
    # File paths
    cme_path = 'Fed_Analysis/FedMeeting_20260128.csv'
    kalshi_decision_path = 'Fed_Analysis/kalshi-price-history-kxfeddecision-26jan-hour.csv'
    kalshi_combo_path = 'Fed_Analysis/kalshi-price-history-kxfedcombo-26jan-hour.csv'

    # Run analysis with equal weights (50/50)
    print("\nRunning analysis with equal weights (50% decision, 50% combo)...")
    analyzer = ForecastComparison(cme_path, kalshi_decision_path, kalshi_combo_path)
    results, comparison = analyzer.run_full_analysis(weight_decision=0.5, weight_combo=0.5)

    return analyzer


if __name__ == "__main__":
    analyzer = main()