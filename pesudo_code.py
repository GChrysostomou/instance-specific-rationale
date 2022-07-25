# if BP on task-level


def func(trial): # define black-box model
    ##### the hyper-parameters for the black-box model ####
    # feature attribute
    # method: topk/contigious
    # length


    ##### the black-box model ####
    model = BERT(load checkpoint) #the trained model
    output_from_bert = model(text_input)
    importance_scores = extract_importance_(feature_attribute = "attention", model, output_from_bert) # the 1st hyper parameters = ['attention', 'gradients',...]
    rationales = extract_rationales_(method = topk,    # the 2nd hyper para
                                    len = 0.2,          # the 3rd hyper para
                                    )
    faithful_scores = evaluate_faithful(rationales, model) # the black-box model output
    return faithful_scores

optimize(func, n_trials=100) # 100 forward pass to update the distribution of expected improvement for different hyper-para
