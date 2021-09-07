from plot_functions import  plot_at_different_N_, plot_radars_, plot_increasing_feat_
from generate_tables import create_table_of_rationale_lengths_, generate_table_for_var_combos_, generate_table_for_divergence_
from generate_tables import generate_time_tables_, make_tables_for_rationale_length_var_


if __name__ == "__main__":

    print("Generating plots and figures")

    # Radar plots

    metric_list =  ["f1 macro avg - model labels", "sufficiency", "comprehensiveness"]

    print("\n1. <s> Plotting Radars")
    plot_radars_(
        metric_list = metric_list
    )

    plot_radars_(
            legend_show= True
    )

    plot_radars_(
        metric_list = metric_list,
        rationale_type= "contigious"
    )

    plot_radars_(
            legend_show= True
    )

    print("1 <e> Radars plotted\n")
    
    print("2.1 <s> Ablation -- MAIN TEXT")
    metric_list =  ["f1 score - model labels", "sufficiency", "comprehensiveness"]

    for metric in metric_list:
        plot_increasing_feat_(
            metric=metric
        )

    plot_increasing_feat_(
            metric=metric,
            plot_legend=True
        )


    print("2.2 <s> Ablation -- Appendix")
    metric_list =  ["f1 score - model labels", "sufficiency", "comprehensiveness"]

    for metric in metric_list:
        plot_increasing_feat_(
            metric=metric,
            you_want_all = True
        )

    plot_increasing_feat_(
            metric=metric,
            you_want_all = True,
            plot_legend=True
        )

    print("2.2 <e> Ablation -- Appendix\n")

    print("3.1 <s> Rationale Length Analysis")
    make_tables_for_rationale_length_var_(rationale_type="topk")
    make_tables_for_rationale_length_var_(rationale_type="contigious")
    print("3.1 <e> Rationale Length Analysis\n")
    print("3.2 <s> Increasing N appendix plots and tables")

    plot_at_different_N_()

    create_table_of_rationale_lengths_(
        divergence="jsd",
        extracted_rationale_dir="extracted_rationales"
    )

    create_table_of_rationale_lengths_(
        divergence="jsd",
        double = True,
        extracted_rationale_dir="extracted_rationales"
    )

    print("3.2 <e> Increasing N appendix plots and tables\n")

    print("4. <s> Different combos")
    
    generate_table_for_var_combos_(rationale_type="topk")
    generate_table_for_var_combos_()

    print("4. <e> Different combos\n")
    print("5. <s> Divegence comparison")

    generate_table_for_divergence_()

    print("6. <s> Tables for time taken")

    generate_time_tables_()

    print("6. <e> Tables for time taken \n")

    print("Success!!")

    




    exit()