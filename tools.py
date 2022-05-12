import os
import paddle

def save_model(model, optimizer, save_dir, epoch, name):
    current_save_dir = os.path.join(save_dir, "epoch_{}".format(epoch))
    if not os.path.isdir(current_save_dir):
        os.makedirs(current_save_dir)
    paddle.save(model.state_dict(),
                os.path.join(current_save_dir, name, 'model.pdparams'))
    paddle.save(optimizer.state_dict(),
                os.path.join(current_save_dir, name, 'model.pdopt'))


def resume(model, optimizer, resume_model):
    if resume_model is not None:
        print('Resume model from {}'.format(resume_model))
        if os.path.exists(resume_model):
            resume_model = os.path.normpath(resume_model)
            ckpt_path = os.path.join(resume_model, 'model.pdparams')
            para_state_dict = paddle.load(ckpt_path)
            ckpt_path = os.path.join(resume_model, 'model.pdopt')
            opti_state_dict = paddle.load(ckpt_path)
            model.set_state_dict(para_state_dict)
            optimizer.set_state_dict(opti_state_dict)

            iter = resume_model.split('_')[-1]
            iter = int(iter)
            return iter
        else:
            raise ValueError(
                'Directory of the model needed to resume is not Found: {}'.
                    format(resume_model))
    else:
        print('No model needed to resume.')