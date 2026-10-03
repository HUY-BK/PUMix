from pathlib import Path
import torch


def save_checkpoint(path, student, teacher, optimizer, scheduler, mspr, spcl, epoch, best_score, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "best_score": best_score,
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "mspr": mspr.state_dict() if mspr is not None else None,
        "spcl": spcl.state_dict() if spcl is not None else None,
        "config": config,
    }, path)


def load_teacher_checkpoint(path, model, device="cpu"):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "teacher" in state:
        model.load_state_dict(state["teacher"], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    return state
