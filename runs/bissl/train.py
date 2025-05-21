import wandb
import torch


def log_data_and_state_dicts(
    args,
    model_p,
    model_paths_dict,
    top1_dbb,
    top5_dbb,
    loss_dbb,
    epoch,
    best_top1_acc,
    best_top5_acc,
    best_acc_top1_prev,
):
    wandb.log(
        {
            "test/top1_acc_test_dbb": top1_dbb,
            "test/top5_acc_test_dbb": top5_dbb,
            "test/best_top1_acc_test_dbb": best_top1_acc,
            "test/best_top5_acc_test_dbb": best_top5_acc,
            "test/avgloss_test_dbb": loss_dbb,
            "test/epoch": epoch + 1,
        }
    )

    if args.save_model:
        model_p.eval()

        sd_p_bb = model_p.backbone.state_dict()

        torch.save(sd_p_bb, model_paths_dict["p_bb"])

        print("Pretext Backbone Saved.")
