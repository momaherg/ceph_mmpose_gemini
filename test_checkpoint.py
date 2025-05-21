#!/usr/bin/env python
import os
import os.path as osp
import sys
from evaluate_checkpoint import evaluate_checkpoint

def test_checkpoint(checkpoint_path, test_json=None, visualize=False):
    """
    Test a model checkpoint using the MRE metric.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        test_json (str, optional): Path to test data JSON
        visualize (bool): Whether to create visualizations
    """
    # Use the current training config
    config_path = 'configs/hrnetv2/hrnetv2_w18_cephalometric_224x224.py'
    
    # Output directories
    out_dir = 'evaluation_results'
    show_dir = 'evaluation_results/visualizations' if visualize else None
    
    # Run evaluation
    print(f"Evaluating checkpoint: {checkpoint_path}")
    print(f"Using config: {config_path}")
    if test_json:
        print(f"Using test data: {test_json}")
    
    try:
        mre_per_keypoint, overall_mre = evaluate_checkpoint(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            test_json=test_json,
            out_dir=out_dir,
            show=False,  # Don't show plots
            show_dir=show_dir  # Save visualizations if requested
        )
        
        print(f"Evaluation complete! Results saved to {out_dir}")
        print(f"Overall MRE: {overall_mre:.4f} pixels")
        
        return {
            'overall_mre': overall_mre,
            'mre_per_keypoint': mre_per_keypoint,
            'results_dir': out_dir
        }
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    # Get checkpoint path from command line
    if len(sys.argv) < 2:
        print("Usage: python test_checkpoint.py <checkpoint_path> [test_json_path] [visualize]")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    test_json = sys.argv[2] if len(sys.argv) > 2 else None
    visualize = (sys.argv[3].lower() == 'true') if len(sys.argv) > 3 else False
    
    test_checkpoint(checkpoint_path, test_json, visualize) 