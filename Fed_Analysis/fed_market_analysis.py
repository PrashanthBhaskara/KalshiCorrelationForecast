"""
Cross-Market Signal Extraction: Fed Decision Markets Analysis

Analyzes two correlated Kalshi markets:
1. Simple Fed decision market (rate changes)
2. Compound market (rate changes + dissent status)

Extracts marginal and conditional probabilities of dissent from compound market.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

class FedMarketAnalysis:
    def __init__(self, decision_path: str, combo_path: str):
        """Initialize with paths to the two market data files."""
        self.decision_path = decision_path
        self.combo_path = combo_path
        self.decision_df = None
        self.combo_df = None
        self.aligned_df = None

    def load_data(self):
        """Load and preprocess both market datasets."""
        print("Loading market data...")

        # Load decision market
        self.decision_df = pd.read_csv(self.decision_path)
        self.decision_df['timestamp'] = pd.to_datetime(self.decision_df['timestamp'])

        # Load combo market
        self.combo_df = pd.read_csv(self.combo_path)
        self.combo_df['timestamp'] = pd.to_datetime(self.combo_df['timestamp'])

        # Convert prices to probabilities (divide by 100 if needed)
        # Check if values are in cents (>1) or already probabilities (<1)
        decision_cols = [c for c in self.decision_df.columns if c != 'timestamp']
        combo_cols = [c for c in self.combo_df.columns if c != 'timestamp']

        # Convert to probabilities (0-1 range)
        for col in decision_cols:
            if self.decision_df[col].max() > 1.5:
                self.decision_df[col] = self.decision_df[col] / 100.0

        for col in combo_cols:
            if self.combo_df[col].max() > 1.5:
                self.combo_df[col] = self.combo_df[col] / 100.0

        print(f"Decision market: {len(self.decision_df)} observations")
        print(f"  Date range: {self.decision_df['timestamp'].min()} to {self.decision_df['timestamp'].max()}")
        print(f"  Outcomes: {decision_cols}")

        print(f"\nCombo market: {len(self.combo_df)} observations")
        print(f"  Date range: {self.combo_df['timestamp'].min()} to {self.combo_df['timestamp'].max()}")
        print(f"  Outcomes: {combo_cols}")

    def align_data(self):
        """Align both datasets by timestamp using forward-fill for missing values."""
        print("\nAligning data by timestamp...")

        # Merge on timestamp with outer join
        self.aligned_df = pd.merge(
            self.decision_df,
            self.combo_df,
            on='timestamp',
            how='outer',
            suffixes=('_decision', '_combo')
        ).sort_values('timestamp')

        # Forward fill missing values (use last known price)
        self.aligned_df = self.aligned_df.ffill()

        # Drop rows with any remaining NaN (early data before both markets started)
        self.aligned_df = self.aligned_df.dropna()

        print(f"Aligned dataset: {len(self.aligned_df)} observations")
        print(f"  Date range: {self.aligned_df['timestamp'].min()} to {self.aligned_df['timestamp'].max()}")

        return self.aligned_df

    def compute_marginal_dissent_probability(self):
        """
        Compute P(Dissent > 0) from combo market.
        Sum all outcomes where dissent > 0.
        """
        print("\nComputing marginal dissent probability...")

        # Find all combo market columns with "Dissents: >0"
        dissent_cols = [col for col in self.aligned_df.columns if 'Dissents: >0' in col]

        # Marginal probability of dissent = sum of all dissent > 0 outcomes
        self.aligned_df['P(Dissent>0)'] = self.aligned_df[dissent_cols].sum(axis=1)
        self.aligned_df['P(Dissent=0)'] = 1 - self.aligned_df['P(Dissent>0)']

        print(f"Found {len(dissent_cols)} outcomes with Dissent > 0:")
        for col in dissent_cols:
            print(f"  - {col}")

        print(f"\nMarginal dissent probability stats:")
        print(f"  Mean: {self.aligned_df['P(Dissent>0)'].mean():.4f}")
        print(f"  Median: {self.aligned_df['P(Dissent>0)'].median():.4f}")
        print(f"  Std: {self.aligned_df['P(Dissent>0)'].std():.4f}")
        print(f"  Range: [{self.aligned_df['P(Dissent>0)'].min():.4f}, {self.aligned_df['P(Dissent>0)'].max():.4f}]")

        return self.aligned_df['P(Dissent>0)']

    def compute_conditional_dissent_probability(self):
        """
        Compute P(Dissent > 0 | Rate decision) for each rate decision.
        Using Bayes: P(D>0 | R) = P(R, D>0) / P(R)
        """
        print("\nComputing conditional dissent probabilities...")

        # Map decision market outcomes to combo market patterns
        decision_mapping = {
            'Fed maintains rate': 'No change',
            'Cut 25bps': '25bp cut',
            'Cut >25bps': '>25bp cut',
            'Hike 25bps': '25bp hike',
            'Hike >25bps': '>25bp hike'
        }

        for decision_col, combo_pattern in decision_mapping.items():
            if decision_col not in self.aligned_df.columns:
                continue

            # Find combo columns matching this decision
            matching_dissent_cols = [col for col in self.aligned_df.columns
                                     if combo_pattern in col and 'Dissents: >0' in col]
            matching_no_dissent_cols = [col for col in self.aligned_df.columns
                                        if combo_pattern in col and 'Dissents: 0' in col]

            if not matching_dissent_cols:
                continue

            # P(R, D>0) = sum of combo outcomes with this rate and dissent > 0
            p_r_and_d = self.aligned_df[matching_dissent_cols].sum(axis=1)

            # P(R) from simple market
            p_r = self.aligned_df[decision_col]

            # P(D>0 | R) = P(R, D>0) / P(R)
            # Avoid division by zero
            conditional_prob = np.where(p_r > 0.001, p_r_and_d / p_r, np.nan)

            col_name = f'P(Dissent>0|{decision_col})'
            self.aligned_df[col_name] = conditional_prob

            # Stats
            valid_probs = conditional_prob[~np.isnan(conditional_prob)]
            if len(valid_probs) > 0:
                print(f"\n{decision_col}:")
                print(f"  Mean P(Dissent>0|{decision_col}): {valid_probs.mean():.4f}")
                print(f"  Median: {np.median(valid_probs):.4f}")
                print(f"  Std: {valid_probs.std():.4f}")

    def check_arbitrage_consistency(self):
        """
        Check if combo market prices are consistent with simple market.
        For each rate decision: sum of combo outcomes for that decision should ≈ simple market price.
        """
        print("\n" + "="*80)
        print("ARBITRAGE CONSISTENCY CHECK")
        print("="*80)

        decision_mapping = {
            'Fed maintains rate': 'No change',
            'Cut 25bps': '25bp cut',
            'Cut >25bps': '>25bp cut',
            'Hike 25bps': '25bp hike',
            'Hike >25bps': '>25bp hike'
        }

        arbitrage_stats = []

        for decision_col, combo_pattern in decision_mapping.items():
            if decision_col not in self.aligned_df.columns:
                continue

            # Find all combo columns for this decision (both dissent statuses)
            matching_cols = [col for col in self.aligned_df.columns if combo_pattern in col and 'Dissents:' in col]

            if not matching_cols:
                continue

            # Sum of combo probabilities for this decision
            combo_sum = self.aligned_df[matching_cols].sum(axis=1)

            # Simple market probability
            simple_prob = self.aligned_df[decision_col]

            # Divergence
            divergence = combo_sum - simple_prob

            # Store divergence
            col_name = f'Divergence_{decision_col}'
            self.aligned_df[col_name] = divergence

            # Statistics
            abs_div = np.abs(divergence)
            stats_dict = {
                'Decision': decision_col,
                'Mean Divergence': divergence.mean(),
                'Mean Abs Divergence': abs_div.mean(),
                'Max Abs Divergence': abs_div.max(),
                'Std Divergence': divergence.std(),
                'RMSE': np.sqrt((divergence**2).mean())
            }
            arbitrage_stats.append(stats_dict)

            print(f"\n{decision_col}:")
            print(f"  Mean divergence: {stats_dict['Mean Divergence']:.6f}")
            print(f"  Mean absolute divergence: {stats_dict['Mean Abs Divergence']:.6f}")
            print(f"  Max absolute divergence: {stats_dict['Max Abs Divergence']:.6f}")
            print(f"  RMSE: {stats_dict['RMSE']:.6f}")

        self.arbitrage_stats_df = pd.DataFrame(arbitrage_stats)
        return self.arbitrage_stats_df

    def visualize_market_prices(self, save_path='Fed_Analysis/plots'):
        """Visualize price evolution for both markets."""
        Path(save_path).mkdir(parents=True, exist_ok=True)

        print("\nGenerating visualizations...")

        # Plot 1: Simple market outcomes over time
        fig, ax = plt.subplots(figsize=(14, 6))

        decision_cols = ['Fed maintains rate', 'Cut 25bps', 'Cut >25bps', 'Hike 25bps', 'Hike >25bps']
        available_cols = [c for c in decision_cols if c in self.aligned_df.columns]

        for col in available_cols:
            ax.plot(self.aligned_df['timestamp'], self.aligned_df[col], label=col, linewidth=2)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title('Fed Decision Market: Rate Change Probabilities Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_path}/fed_decision_prices.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}/fed_decision_prices.png")
        plt.close()

        # Plot 2: Combo market outcomes
        fig, ax = plt.subplots(figsize=(14, 6))

        combo_cols = [c for c in self.aligned_df.columns if 'Federal Funds Rates:' in c]
        for col in combo_cols:
            ax.plot(self.aligned_df['timestamp'], self.aligned_df[col], label=col, alpha=0.7)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title('Fed Combo Market: Rate + Dissent Combinations Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_path}/fed_combo_prices.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}/fed_combo_prices.png")
        plt.close()

        # Plot 3: Marginal dissent probability
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(self.aligned_df['timestamp'], self.aligned_df['P(Dissent>0)'],
                linewidth=2, color='darkred', label='P(Dissent > 0)')
        ax.axhline(y=self.aligned_df['P(Dissent>0)'].mean(), color='red',
                   linestyle='--', alpha=0.5, label=f'Mean: {self.aligned_df["P(Dissent>0)"].mean():.3f}')

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title('Extracted Signal: Marginal Probability of Dissenting Votes', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_path}/marginal_dissent_probability.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}/marginal_dissent_probability.png")
        plt.close()

        # Plot 4: Conditional dissent probabilities
        fig, ax = plt.subplots(figsize=(14, 6))

        conditional_cols = [c for c in self.aligned_df.columns if 'P(Dissent>0|' in c]
        for col in conditional_cols:
            decision_name = col.split('|')[1].rstrip(')')
            ax.plot(self.aligned_df['timestamp'], self.aligned_df[col],
                   label=f'Given {decision_name}', linewidth=2, alpha=0.8)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Conditional Probability', fontsize=12)
        ax.set_title('Conditional Probability of Dissent Given Rate Decision', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_path}/conditional_dissent_probabilities.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}/conditional_dissent_probabilities.png")
        plt.close()

    def visualize_arbitrage(self, save_path='Fed_Analysis/plots'):
        """Visualize arbitrage divergence over time."""
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # Find divergence columns
        divergence_cols = [c for c in self.aligned_df.columns if c.startswith('Divergence_')]

        if not divergence_cols:
            print("No divergence data to plot")
            return

        fig, axes = plt.subplots(len(divergence_cols), 1, figsize=(14, 4*len(divergence_cols)))

        if len(divergence_cols) == 1:
            axes = [axes]

        for ax, col in zip(axes, divergence_cols):
            decision_name = col.replace('Divergence_', '')

            ax.plot(self.aligned_df['timestamp'], self.aligned_df[col],
                   linewidth=1.5, color='purple', alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.axhline(y=self.aligned_df[col].mean(), color='red',
                      linestyle='--', alpha=0.5, label=f'Mean: {self.aligned_df[col].mean():.6f}')

            ax.set_ylabel('Divergence', fontsize=11)
            ax.set_title(f'Arbitrage Divergence: {decision_name}', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        axes[-1].set_xlabel('Date', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{save_path}/arbitrage_divergence.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}/arbitrage_divergence.png")
        plt.close()

    def statistical_tests(self):
        """Perform statistical tests on the extracted signals."""
        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS")
        print("="*80)

        # Test 1: Is marginal dissent probability significantly > 0?
        dissent_prob = self.aligned_df['P(Dissent>0)'].dropna()
        t_stat, p_value = stats.ttest_1samp(dissent_prob, 0)

        print(f"\n1. One-sample t-test: Is P(Dissent>0) significantly different from 0?")
        print(f"   t-statistic: {t_stat:.4f}")
        print(f"   p-value: {p_value:.6f}")
        print(f"   Result: {'Yes' if p_value < 0.05 else 'No'} (at α=0.05)")

        # Test 2: Time trend in dissent probability
        time_idx = np.arange(len(dissent_prob))
        slope, intercept, r_value, p_value_trend, std_err = stats.linregress(time_idx, dissent_prob)

        print(f"\n2. Linear regression: Time trend in P(Dissent>0)")
        print(f"   Slope: {slope:.6f} per hour")
        print(f"   R²: {r_value**2:.4f}")
        print(f"   p-value: {p_value_trend:.6f}")
        print(f"   Trend: {'Significant' if p_value_trend < 0.05 else 'Not significant'} (at α=0.05)")

        # Test 3: Arbitrage divergence tests
        print(f"\n3. Arbitrage divergence: Are markets consistent?")
        for col in [c for c in self.aligned_df.columns if c.startswith('Divergence_')]:
            div = self.aligned_df[col].dropna()
            t_stat, p_value = stats.ttest_1samp(div, 0)

            decision_name = col.replace('Divergence_', '')
            print(f"\n   {decision_name}:")
            print(f"     Mean divergence: {div.mean():.6f}")
            print(f"     t-statistic: {t_stat:.4f}")
            print(f"     p-value: {p_value:.6f}")
            print(f"     Result: {'Significant bias' if p_value < 0.05 else 'No significant bias'} (at α=0.05)")

        # Test 4: Volatility comparison
        print(f"\n4. Volatility comparison:")
        simple_volatility = self.aligned_df['Fed maintains rate'].std()
        marginal_volatility = self.aligned_df['P(Dissent>0)'].std()

        print(f"   Simple market volatility (Fed maintains rate): {simple_volatility:.6f}")
        print(f"   Extracted signal volatility (P(Dissent>0)): {marginal_volatility:.6f}")
        print(f"   Ratio: {marginal_volatility / simple_volatility:.2f}x")

    def save_results(self, output_path='Fed_Analysis/fed_analysis_results.csv'):
        """Save the aligned dataset with all computed probabilities."""
        self.aligned_df.to_csv(output_path, index=False)
        print(f"\n✅ Saved full analysis results to: {output_path}")

        # Also save summary statistics
        summary_path = output_path.replace('.csv', '_summary.csv')
        if hasattr(self, 'arbitrage_stats_df'):
            self.arbitrage_stats_df.to_csv(summary_path, index=False)
            print(f"✅ Saved arbitrage summary to: {summary_path}")

    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("="*80)
        print("FED DECISION MARKETS: CROSS-MARKET SIGNAL EXTRACTION ANALYSIS")
        print("="*80)

        self.load_data()
        self.align_data()
        self.compute_marginal_dissent_probability()
        self.compute_conditional_dissent_probability()
        self.check_arbitrage_consistency()
        self.visualize_market_prices()
        self.visualize_arbitrage()
        self.statistical_tests()
        self.save_results()

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)

        return self.aligned_df


def main():
    # File paths
    decision_path = 'Fed_Analysis/kalshi-price-history-kxfeddecision-26jan-hour.csv'
    combo_path = 'Fed_Analysis/kalshi-price-history-kxfedcombo-26jan-hour.csv'

    # Run analysis
    analyzer = FedMarketAnalysis(decision_path, combo_path)
    results = analyzer.run_full_analysis()

    return analyzer


if __name__ == "__main__":
    analyzer = main()
