import data_loader
import evaluator
import config
import utils

def display_top_features(feature_names, importance_scores, modality_name, num_features=10):
    """Display the top N features with their importance scores for a given modality."""
    if not feature_names or importance_scores is None:
        return
    
    sorted_features = sorted(zip(feature_names, importance_scores), 
                           key=lambda x: x[1], reverse=True)
    
    print(f"\n{modality_name} Top {num_features} Features:")
    for index, (name, score) in enumerate(sorted_features[:num_features], 1):
        print(f"{index}. {name}: {score:.3f}")

def execute_evaluations(model_evaluator, feature_sets, target_labels, participant_ids, model_type, scenes, tasks, enable_scene_printing):
    """
    Execute all evaluation methods with clear commenting for enabling/disabling.
    Returns multimodal importance scores for feature analysis.
    """
    # Unpack feature sets for clarity
    audio_features, motion_features, facial_features = feature_sets
    combined_feature_set = (audio_features, motion_features, facial_features)
    
    # ======================================================
    # Evaluation Methods (Enable/disable by commenting)
    # ======================================================
    if 'audio' in tasks:
        # 1. Audio modality evaluation
        model_evaluator.evaluate_feature_combination(
            audio_features, target_labels, scenes, "Audio", participant_ids, model_type, enable_scene_printing
        )
    if 'motion' in tasks:
        # 2. Motion modality evaluation
        model_evaluator.evaluate_feature_combination(
            motion_features, target_labels, scenes, "Motion", participant_ids, model_type, enable_scene_printing
        )
    if 'face' in tasks:
        # 3. Facial modality evaluation
        model_evaluator.evaluate_feature_combination(
            facial_features, target_labels, scenes, "Face", participant_ids, model_type, enable_scene_printing
        )
    if 'late_fusion' in tasks:
        # 4. Decision-level fusion evaluation
        model_evaluator.evaluate_multimodal_fusion(
            combined_feature_set, 
            target_labels, 
            "Multimodal-Late Fusion (Stacking)", 
            participant_ids, 
            model_type, 
            model_type, 
            model_type
        )
    if 'early_fusion' in tasks:
        # 5. Feature-level fusion evaluation
        multimodal_importance_scores, indicator_indices  = model_evaluator.evaluate_feature_combination(
            combined_feature_set, 
            target_labels, 
            scenes,
            "Multimodal-Early Fusion (Concatenation)", 
            participant_ids, 
            model_type,
            enable_scene_printing
        )
        return multimodal_importance_scores, indicator_indices

    return None, None

def main():
    """Main function to orchestrate the experiment workflow."""
    enable_feature_printing = False
    enable_scene_printing = False
    utils.initialize_random_seed(config.seed)
    
    # Load dataset
    participant_ids, feature_sets, feature_names, labels, scenes = data_loader.load_data(enable_scene_printing)
    
    # Initialize model evaluator
    model_evaluator = evaluator.ModelEvaluator()

    tasks = [
        'audio', 
        'motion', 
        'face', 
        'early_fusion', 
        'late_fusion',
        ]
    # Execute evaluations (modify evaluation methods in execute_evaluations as needed)
    multimodal_importance_scores, indicator_indices = execute_evaluations(
        model_evaluator, 
        feature_sets,
        labels,
        participant_ids,
        config.model_type,
        scenes,
        tasks=tasks,
        enable_scene_printing=enable_scene_printing,
    )
    
    print("\n=== Evaluation Completed ===")
    
    if config.model_type in ['lgbm', 'rf'] and enable_feature_printing and multimodal_importance_scores is not None:
        all_original_feature_names = sum(feature_names, [])
        final_feature_names = all_original_feature_names[:]        

        if indicator_indices.size > 0:
            actual_indicator_names = [
                f"{all_original_feature_names[i]}_indicator" 
                for i in indicator_indices
            ]
            final_feature_names = all_original_feature_names + actual_indicator_names

        display_top_features(
            final_feature_names,
            multimodal_importance_scores, 
            "Multimodal", 
            num_features=20
        )

if __name__ == "__main__":
    main()