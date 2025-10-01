"""
Test Enhanced Coprime Processing for Literature Claim Validation

This test validates whether the enhanced coprime processing with full-period
coherent processing and proper matched filtering achieves the literature-
claimed 2-3 dB SNR improvement over standard waveforms.
"""

import numpy as np
import matplotlib.pyplot as plt
from enhanced_coprime_processing import create_enhanced_coprime_processor


def test_coprime_snr_improvement():
    """Test coprime SNR improvement across multiple SNR levels."""
    print("="*70)
    print("ENHANCED COPRIME PROCESSING VALIDATION")
    print("="*70)
    print("Goal: Validate literature claim of 2-3 dB SNR improvement")

    # Test parameters
    true_angles = np.array([-6, 6])  # Moderate separation for challenging test
    n_sensors = 8
    snr_range = np.arange(4, 12, 1)  # Focus on practical SNR range
    coprime_pairs = [(31, 37), (13, 17), (7, 11)]  # Different coprime configurations

    print(f"Test configuration:")
    print(f"  True angles: {true_angles}")
    print(f"  Array sensors: {n_sensors}")
    print(f"  SNR range: {snr_range[0]} to {snr_range[-1]} dB")
    print(f"  Coprime pairs: {coprime_pairs}")

    results = {}

    for coprime_pair in coprime_pairs:
        print(f"\\n[TESTING COPRIME PAIR: {coprime_pair}]")

        # Create enhanced coprime processor
        processor = create_enhanced_coprime_processor(coprime_pair, n_sensors)

        pair_results = []

        for snr_db in snr_range:
            print(f"  SNR = {snr_db} dB: ", end="", flush=True)

            # Test with Enhanced SPICE (our best algorithm)
            comparison = processor.compare_coprime_vs_standard(
                true_angles, snr_db, use_enhanced_spice=True
            )

            # Extract key metrics
            coprime_perf = comparison['coprime_performance']
            standard_perf = comparison['standard_performance']
            improvements = comparison['improvements']
            incoherence = comparison['incoherence_analysis']

            result = {
                'snr_db': snr_db,
                'coprime_detection_rate': coprime_perf['detection_rate'],
                'standard_detection_rate': standard_perf['detection_rate'],
                'detection_improvement': improvements['detection_improvement'],
                'angular_improvement': improvements['angular_improvement'],
                'false_alarm_improvement': improvements['false_alarm_improvement'],
                'coherence_improvement': incoherence['coherence_improvement'],
                'condition_number_improvement': incoherence['condition_number_ratio'],
                'processing_details': comparison['processing_details']
            }

            pair_results.append(result)

            print(f"Det={improvements['detection_improvement']:.2f}x, " +
                  f"Coh={incoherence['coherence_improvement']:.2f}x")

        results[f"coprime_{coprime_pair[0]}_{coprime_pair[1]}"] = pair_results

    return results


def analyze_coprime_improvements(results: dict) -> dict:
    """Analyze coprime improvements to validate literature claims."""
    print(f"\\n[COPRIME IMPROVEMENT ANALYSIS]")

    analysis = {}

    for pair_name, pair_results in results.items():
        print(f"\\n{pair_name.replace('_', ' ').title()}:")

        # Compute average improvements across SNR range
        detection_improvements = [r['detection_improvement'] for r in pair_results]
        angular_improvements = [r['angular_improvement'] for r in pair_results]
        coherence_improvements = [r['coherence_improvement'] for r in pair_results]

        avg_detection = np.mean(detection_improvements)
        avg_angular = np.mean(angular_improvements)
        avg_coherence = np.mean(coherence_improvements)

        # Find best case improvements
        max_detection = np.max(detection_improvements)
        max_angular = np.max(angular_improvements)
        max_coherence = np.max(coherence_improvements)

        print(f"  Average detection improvement: {avg_detection:.2f}x")
        print(f"  Average angular improvement: {avg_angular:.2f}x")
        print(f"  Average coherence improvement: {avg_coherence:.2f}x")
        print(f"  Best detection improvement: {max_detection:.2f}x")

        # Convert detection improvement to equivalent SNR improvement
        # Rule of thumb: 2x detection improvement ≈ 3 dB SNR improvement
        equivalent_snr_improvement = 3 * np.log2(avg_detection)

        print(f"  Equivalent SNR improvement: {equivalent_snr_improvement:.1f} dB")

        analysis[pair_name] = {
            'avg_detection_improvement': avg_detection,
            'avg_angular_improvement': avg_angular,
            'avg_coherence_improvement': avg_coherence,
            'max_detection_improvement': max_detection,
            'equivalent_snr_improvement_db': equivalent_snr_improvement,
            'individual_results': pair_results
        }

    # Overall assessment
    print(f"\\n[LITERATURE CLAIM VALIDATION]")

    best_pair = max(analysis.keys(),
                   key=lambda k: analysis[k]['equivalent_snr_improvement_db'])

    best_improvement = analysis[best_pair]['equivalent_snr_improvement_db']
    best_detection = analysis[best_pair]['avg_detection_improvement']

    print(f"Best performing coprime pair: {best_pair}")
    print(f"Best equivalent SNR improvement: {best_improvement:.1f} dB")
    print(f"Best detection improvement: {best_detection:.2f}x")

    # Validate against literature claims
    if best_improvement >= 2.0:
        print(f"[VALIDATED] Literature claim ACHIEVED!")
        print(f"  Claimed: 2-3 dB improvement")
        print(f"  Achieved: {best_improvement:.1f} dB improvement")
    elif best_improvement >= 1.0:
        print(f"[PARTIAL] Partial validation achieved:")
        print(f"  Achieved: {best_improvement:.1f} dB improvement")
        print(f"  Target: 2-3 dB improvement")
    else:
        print(f"[NOT VALIDATED] Literature claim not achieved:")
        print(f"  Achieved: {best_improvement:.1f} dB improvement")
        print(f"  Target: 2-3 dB improvement")

    return analysis


def plot_coprime_performance(results: dict, analysis: dict):
    """Plot coprime performance comparison."""
    plt.figure(figsize=(16, 12))

    # Plot 1: Detection improvement vs SNR
    plt.subplot(2, 3, 1)
    colors = ['blue', 'red', 'green']
    for i, (pair_name, pair_results) in enumerate(results.items()):
        snr_values = [r['snr_db'] for r in pair_results]
        detection_improvements = [r['detection_improvement'] for r in pair_results]

        plt.plot(snr_values, detection_improvements, 'o-',
                color=colors[i % len(colors)], linewidth=2,
                label=pair_name.replace('_', ' ').title())

    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='No improvement')
    plt.axhline(y=1.6, color='orange', linestyle=':', alpha=0.7, label='~2 dB equivalent')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Detection Improvement Factor')
    plt.title('Detection Rate Improvement vs SNR')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 2: Coherence improvement vs SNR
    plt.subplot(2, 3, 2)
    for i, (pair_name, pair_results) in enumerate(results.items()):
        snr_values = [r['snr_db'] for r in pair_results]
        coherence_improvements = [r['coherence_improvement'] for r in pair_results]

        plt.plot(snr_values, coherence_improvements, 'o-',
                color=colors[i % len(colors)], linewidth=2,
                label=pair_name.replace('_', ' ').title())

    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='No improvement')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Coherence Improvement Factor')
    plt.title('Mutual Coherence Improvement vs SNR')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 3: Equivalent SNR improvement by coprime pair
    plt.subplot(2, 3, 3)
    pair_names = list(analysis.keys())
    snr_improvements = [analysis[pair]['equivalent_snr_improvement_db'] for pair in pair_names]

    bars = plt.bar(range(len(pair_names)), snr_improvements,
                  color=['blue', 'red', 'green'][:len(pair_names)])
    plt.axhline(y=2.0, color='orange', linestyle='--', alpha=0.7, label='Literature Claim (2 dB)')
    plt.axhline(y=3.0, color='red', linestyle=':', alpha=0.7, label='Literature Claim (3 dB)')

    plt.xlabel('Coprime Pair')
    plt.ylabel('Equivalent SNR Improvement (dB)')
    plt.title('Equivalent SNR Improvement by Coprime Pair')
    plt.xticks(range(len(pair_names)), [name.replace('_', ' ').title() for name in pair_names], rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add value labels on bars
    for bar, value in zip(bars, snr_improvements):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.1f} dB', ha='center', va='bottom')

    # Plot 4: Detection rate comparison at different SNRs
    plt.subplot(2, 3, 4)
    # Use first coprime pair for detailed comparison
    first_pair_results = list(results.values())[0]
    snr_values = [r['snr_db'] for r in first_pair_results]
    coprime_rates = [r['coprime_detection_rate'] for r in first_pair_results]
    standard_rates = [r['standard_detection_rate'] for r in first_pair_results]

    plt.plot(snr_values, standard_rates, 'b-o', linewidth=2, label='Standard Processing')
    plt.plot(snr_values, coprime_rates, 'r-o', linewidth=2, label='Coprime Processing')

    plt.xlabel('SNR (dB)')
    plt.ylabel('Detection Rate')
    plt.title('Detection Rate: Standard vs Coprime')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1.05)

    # Plot 5: Processing benefits breakdown
    plt.subplot(2, 3, 5)
    best_pair_name = max(analysis.keys(),
                        key=lambda k: analysis[k]['equivalent_snr_improvement_db'])
    best_analysis = analysis[best_pair_name]

    metrics = ['Detection\\nImprovement', 'Angular\\nImprovement', 'Coherence\\nImprovement']
    values = [best_analysis['avg_detection_improvement'],
             best_analysis['avg_angular_improvement'],
             best_analysis['avg_coherence_improvement']]

    bars = plt.bar(metrics, values, color=['green', 'blue', 'orange'])
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    plt.ylabel('Improvement Factor')
    plt.title(f'Processing Benefits\\n({best_pair_name.replace("_", " ").title()})')
    plt.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.2f}x', ha='center', va='bottom')

    # Plot 6: Full period processing impact
    plt.subplot(2, 3, 6)
    # Show processing details for first pair
    first_pair_results = list(results.values())[0]
    processing_lengths = [r['processing_details']['n_snapshots'] for r in first_pair_results]
    improvements = [r['detection_improvement'] for r in first_pair_results]

    plt.scatter(processing_lengths, improvements, c=snr_values, cmap='viridis', s=100)
    plt.colorbar(label='SNR (dB)')
    plt.xlabel('Processing Length (snapshots)')
    plt.ylabel('Detection Improvement Factor')
    plt.title('Improvement vs Processing Length')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('enhanced_coprime_validation.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main validation function."""
    print("Testing Enhanced Coprime Processing for Literature Claim Validation...")

    # Run coprime SNR improvement tests
    results = test_coprime_snr_improvement()

    # Analyze improvements
    analysis = analyze_coprime_improvements(results)

    # Plot results
    plot_coprime_performance(results, analysis)

    print(f"\\n" + "="*70)
    print("COPRIME VALIDATION COMPLETE")
    print("="*70)

    return results, analysis


if __name__ == "__main__":
    results, analysis = main()