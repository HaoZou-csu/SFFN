import numpy as np
import torch
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from utils.util import Scaler, DummyScaler

from time import time
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item(), len(target))
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target), len(target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target), len(target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)



class GraghTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (input, target, _) in enumerate(self.data_loader):
            x_1 = input[0].to(self.device)
            x_2 = input[1].to(self.device)
            x_3 = input[2].to(self.device)
            # x_4 = torch.tensor(input[3]).to(device)
            x_4 = input[3]
            target = target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x_1, x_2, x_3, x_4)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()


            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item(), len(target))
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target), len(target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (input, target, _) in enumerate(self.valid_data_loader):
                x_1 = input[0].to(self.device)
                x_2 = input[1].to(self.device)
                x_3 = input[2].to(self.device)
                # x_4 = torch.tensor(input[3]).to(device)
                x_4 = input[3]
                target = target.to(self.device)

                output = self.model(x_1, x_2, x_3, x_4)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item(), len(target))
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target), len(target))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)




class MMTrainer(GraghTrainer):
    def __int__(self):
        super().__int__()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, ((input_1, input_2, input_3, input_4), text, mag, target, idx) in tqdm(enumerate(self.data_loader)):

            target = target.to(self.device)

            input_1 = input_1.to(self.device)
            input_2 = input_2.to(self.device)
            input_3 = input_3.to(self.device)

            mag = torch.from_numpy(np.array(mag)).to(self.device).to(torch.float32)

            self.optimizer.zero_grad()
            output = self.model(input_1, input_2, input_3, input_4, text, mag)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()


            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item(), len(target))
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target), len(target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx,  ((input_1, input_2, input_3, input_4), text, mag, target, idx) in enumerate(self.valid_data_loader):
                target = target.to(self.device)

                input_1 = input_1.to(self.device)
                input_2 = input_2.to(self.device)
                input_3 = input_3.to(self.device)

                mag = torch.from_numpy(np.array(mag)).to(self.device).to(torch.float32)

                output = self.model(input_1, input_2, input_3, input_4 ,text, mag)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item(), len(target))
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target), len(target))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()


class CombinedTrainer(GraghTrainer):
    def __int__(self):
        super().__int__()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, ((roost_elem_weights, roost_elem_fea, roost_self_idx, roost_nbr_idx, roost_crystal_elem_idx),
        roost_targets,
        roost_ids,
        crabnet_src,
        crabnet_targets,
        crabnet_ids,
        magpie_fea,
        meredig_fea,
        rsc_fea,
        ec_fea) in tqdm(enumerate(self.data_loader)):
            roost_elem_weights = roost_elem_weights.to(self.device)
            roost_elem_fea = roost_elem_fea.to(self.device)
            roost_self_idx = roost_self_idx.to(self.device)
            roost_nbr_idx = roost_nbr_idx.to(self.device)
            roost_crystal_elem_idx = roost_crystal_elem_idx.to(self.device)

            crabnet_src = crabnet_src.to(self.device)

            src, frac = crabnet_src.squeeze(-1).chunk(2, dim=1)
            frac = frac * (1 + (torch.randn_like(frac)) * 0.02)  # normal
            frac = torch.clamp(frac, 0, 1)
            frac[src == 0] = 0
            frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])
            src = src.long()
            frac = frac.float()

            src, frac = src.to(self.device), frac.to(self.device)
            targets = crabnet_targets.to(self.device).view(-1,1)

            magpie_fea = magpie_fea.to(self.device)
            meredig_fea = meredig_fea.to(self.device)
            rsc_fea = rsc_fea.to(self.device)
            ec_fea = ec_fea.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(roost_elem_weights, roost_elem_fea, roost_self_idx, roost_nbr_idx, roost_crystal_elem_idx, src, frac, magpie_fea, meredig_fea, rsc_fea, ec_fea)


            loss = self.criterion(output, targets)
            loss.backward()
            self.optimizer.step()


            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item(), len(targets))
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, targets), len(targets))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        for key, value in log.items():
            self.writer.add_scalar(f'train_{key}', value, epoch)

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            for key, value in val_log.items():
                self.writer.add_scalar(f'val_{key}', value, epoch)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, ((roost_elem_weights, roost_elem_fea, roost_self_idx, roost_nbr_idx, roost_crystal_elem_idx),
            roost_targets,
            roost_ids,
            crabnet_src,
            crabnet_targets,
            crabnet_ids,
            magpie_fea,
            meredig_fea,
            rsc_fea,
            ec_fea) in tqdm(enumerate(self.valid_data_loader)):
                roost_elem_weights = roost_elem_weights.to(self.device)
                roost_elem_fea = roost_elem_fea.to(self.device)
                roost_self_idx = roost_self_idx.to(self.device)
                roost_nbr_idx = roost_nbr_idx.to(self.device)
                roost_crystal_elem_idx = roost_crystal_elem_idx.to(self.device)

                crabnet_src = crabnet_src.to(self.device)

                src, frac = crabnet_src.squeeze(-1).chunk(2, dim=1)
                frac = frac * (1 + (torch.randn_like(frac)) * 0.02)
                frac = torch.clamp(frac, 0, 1)
                frac[src == 0] = 0
                frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])
                src = src.long()
                frac = frac.float()

                src, frac = src.to(self.device), frac.to(self.device)
                targets = crabnet_targets.to(self.device).view(-1,1)

                magpie_fea = magpie_fea.to(self.device)
                meredig_fea = meredig_fea.to(self.device)
                rsc_fea = rsc_fea.to(self.device)
                ec_fea = ec_fea.to(self.device)

                output = self.model(roost_elem_weights, roost_elem_fea, roost_self_idx, roost_nbr_idx, roost_crystal_elem_idx, src, frac, magpie_fea, meredig_fea, rsc_fea, ec_fea)
                loss = self.criterion(output, targets)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item(), len(targets))
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, targets), len(targets))
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def test(self, test_loader):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, ((roost_elem_weights, roost_elem_fea, roost_self_idx, roost_nbr_idx, roost_crystal_elem_idx),
                            roost_targets,
                            roost_ids,
                            crabnet_src,
                            crabnet_targets,
                            crabnet_ids,
                            magpie_fea,
                            meredig_fea,
                            rsc_fea,
                            ec_fea) in tqdm(enumerate(test_loader)):
                roost_elem_weights = roost_elem_weights.to(self.device)
                roost_elem_fea = roost_elem_fea.to(self.device)
                roost_self_idx = roost_self_idx.to(self.device)
                roost_nbr_idx = roost_nbr_idx.to(self.device)
                roost_crystal_elem_idx = roost_crystal_elem_idx.to(self.device)

                crabnet_src = crabnet_src.to(self.device)

                src, frac = crabnet_src.squeeze(-1).chunk(2, dim=1)
                frac = frac * (1 + (torch.randn_like(frac)) * 0.02)
                frac = torch.clamp(frac, 0, 1)
                frac[src == 0] = 0
                frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])
                src = src.long()
                frac = frac.float()

                src, frac = src.to(self.device), frac.to(self.device)
                targets = crabnet_targets.to(self.device).view(-1, 1)

                magpie_fea = magpie_fea.to(self.device)
                meredig_fea = meredig_fea.to(self.device)
                rsc_fea = rsc_fea.to(self.device)
                ec_fea = ec_fea.to(self.device)

                output = self.model(roost_elem_weights, roost_elem_fea, roost_self_idx, roost_nbr_idx, roost_crystal_elem_idx, src, frac, magpie_fea, meredig_fea, rsc_fea, ec_fea)
                all_preds.append(output.cpu())
                all_targets.append(targets.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        mae = mean_absolute_error(all_targets, all_preds)
        rmse = mean_squared_error(all_targets, all_preds) ** 0.5

        # print(f"Test Accuracy: {accuracy}")
        # print(f"Test AUC: {auc}")
        print(f'Test MAE: {mae}')
        print(f'Test RMSE: {rmse}')

        return mae, rmse

class CrabnetTrainer(GraghTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, data_loader, valid_data_loader=None,
                 lr_scheduler=None, classification=False, *args, **kwargs):
        # Initialization without the 'classification' argument
        super().__init__(model, criterion, metric_ftns, optimizer, config, device, data_loader, valid_data_loader,
                         lr_scheduler, **kwargs)
        # Set classification flag if needed
        self.classification = classification
        # Initialize scaler
        train_y = np.load('data/MP/y_train.npy')
        self.scaler = DummyScaler(train_y) if classification else Scaler(train_y)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, ((roost_elem_weights, roost_elem_fea, roost_self_idx, roost_nbr_idx, roost_crystal_elem_idx),
        roost_targets,
        roost_ids,
        crabnet_src,
        crabnet_targets,
        crabnet_ids,
        magpie_fea,
        meredig_fea,
        rsc_fea,
        ec_fea) in tqdm(enumerate(self.data_loader)):

            crabnet_src = crabnet_src.to(self.device)

            src, frac = crabnet_src.squeeze(-1).chunk(2, dim=1)
            frac = frac * (1 + (torch.randn_like(frac)) * 0.02)  # normal
            frac = torch.clamp(frac, 0, 1)
            frac[src == 0] = 0
            frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])
            src = src.long()
            frac = frac.float()

            src, frac = src.to(self.device), frac.to(self.device)
            targets = crabnet_targets.to(self.device).view(-1,1)
            targets = self.scaler.scale(targets)

            self.optimizer.zero_grad()
            output = self.model(src, frac)
            output, uncertainty = output.chunk(2, dim=-1)

            loss = self.criterion(output, targets)
            loss.backward()
            self.optimizer.step()


            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item(), len(targets))
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, targets), len(targets))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        for key, value in log.items():
            self.writer.add_scalar(f'train_{key}', value, epoch)

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            for key, value in val_log.items():
                self.writer.add_scalar(f'val_{key}', value, epoch)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, ((roost_elem_weights, roost_elem_fea, roost_self_idx, roost_nbr_idx, roost_crystal_elem_idx),
            roost_targets,
            roost_ids,
            crabnet_src,
            crabnet_targets,
            crabnet_ids,
            magpie_fea,
            meredig_fea,
            rsc_dea,
            ec_fea) in tqdm(enumerate(self.valid_data_loader)):

                crabnet_src = crabnet_src.to(self.device)

                src, frac = crabnet_src.squeeze(-1).chunk(2, dim=1)
                frac = frac * (1 + (torch.randn_like(frac)) * 0.02)
                frac = torch.clamp(frac, 0, 1)
                frac[src == 0] = 0
                frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])
                src = src.long()
                frac = frac.float()

                src, frac = src.to(self.device), frac.to(self.device)
                targets = crabnet_targets.to(self.device).view(-1,1)

                output = self.model(src, frac)
                output, uncertainty = output.chunk(2, dim=-1)
                output = self.scaler.unscale(output)

                loss = self.criterion(output, targets)
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item(), len(targets))
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, targets), len(targets))
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()


class SingleTrainer(GraghTrainer):
    def __int__(self):
        super().__int__()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, ((roost_elem_weights, roost_elem_fea, roost_self_idx, roost_nbr_idx, roost_crystal_elem_idx),
        roost_targets,
        roost_ids,
        crabnet_src,
        crabnet_targets,
        crabnet_ids,
        magpie_fea,
        meredig_fea,
        rsc_fea,
        ec_fea) in tqdm(enumerate(self.data_loader)):

            targets = crabnet_targets.to(self.device).view(-1,1)

            magpie_fea = magpie_fea.to(self.device)
            meredig_fea = meredig_fea.to(self.device)
            rsc_fea = rsc_fea.to(self.device)
            ec_fea = ec_fea.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(magpie_fea)

            loss = self.criterion(output, targets)
            loss.backward()
            self.optimizer.step()


            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item(), len(targets))
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, targets), len(targets))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        for key, value in log.items():
            self.writer.add_scalar(f'train_{key}', value, epoch)

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            for key, value in val_log.items():
                self.writer.add_scalar(f'val_{key}', value, epoch)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, ((roost_elem_weights, roost_elem_fea, roost_self_idx, roost_nbr_idx, roost_crystal_elem_idx),
            roost_targets,
            roost_ids,
            crabnet_src,
            crabnet_targets,
            crabnet_ids,
            magpie_fea,
            meredig_fea,
            rsc_fea,
            ec_fea) in tqdm(enumerate(self.valid_data_loader)):

                targets = crabnet_targets.to(self.device).view(-1,1)

                magpie_fea = magpie_fea.to(self.device)
                meredig_fea = meredig_fea.to(self.device)
                rsc_fea = rsc_fea.to(self.device)
                ec_fea = ec_fea.to(self.device)

                output = self.model(magpie_fea)
                loss = self.criterion(output, targets)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item(), len(targets))
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, targets), len(targets))
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def test(self, test_loader):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, ((roost_elem_weights, roost_elem_fea, roost_self_idx, roost_nbr_idx, roost_crystal_elem_idx),
                            roost_targets,
                            roost_ids,
                            crabnet_src,
                            crabnet_targets,
                            crabnet_ids,
                            magpie_fea,
                            meredig_fea,
                            rsc_fea,
                            ec_fea) in tqdm(enumerate(test_loader)):

                targets = crabnet_targets.to(self.device).view(-1, 1)

                magpie_fea = magpie_fea.to(self.device)
                meredig_fea = meredig_fea.to(self.device)
                rsc_fea = rsc_fea.to(self.device)
                ec_fea = ec_fea.to(self.device)

                output = self.model(magpie_fea)
                all_preds.append(output.cpu())
                all_targets.append(targets.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        mae = mean_absolute_error(all_targets, all_preds)
        rmse = mean_squared_error(all_targets, all_preds) ** 0.5

        # print(f"Test Accuracy: {accuracy}")
        # print(f"Test AUC: {auc}")
        print(f'Test MAE: {mae}')
        print(f'Test RMSE: {rmse}')

        return mae, rmse

class RoostTrainer(GraghTrainer):
    def __int__(self):
        super().__int__()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, ((roost_elem_weights, roost_elem_fea, roost_self_idx, roost_nbr_idx, roost_crystal_elem_idx),
                        roost_targets,
                        roost_ids,
                        crabnet_src,
                        crabnet_targets,
                        crabnet_ids,
                        magpie_fea,
                        meredig_fea,
                        rsc_fea,
                        ec_fea) in tqdm(enumerate(self.data_loader)):
            roost_elem_weights = roost_elem_weights.to(self.device)
            roost_elem_fea = roost_elem_fea.to(self.device)
            roost_self_idx = roost_self_idx.to(self.device)
            roost_nbr_idx = roost_nbr_idx.to(self.device)
            roost_crystal_elem_idx = roost_crystal_elem_idx.to(self.device)

            crabnet_src = crabnet_src.to(self.device)

            targets = crabnet_targets.to(self.device).view(-1, 1)
            #
            magpie_fea = magpie_fea.to(self.device)
            meredig_fea = meredig_fea.to(self.device)
            rsc_fea = rsc_fea.to(self.device)
            ec_fea = ec_fea.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(roost_elem_weights, roost_elem_fea, roost_self_idx, roost_nbr_idx,
                                roost_crystal_elem_idx)

            loss = self.criterion(output[0], targets)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item(), len(targets))
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output[0], targets), len(targets))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        for key, value in log.items():
            self.writer.add_scalar(f'train_{key}', value, epoch)

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            for key, value in val_log.items():
                self.writer.add_scalar(f'val_{key}', value, epoch)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (
            (roost_elem_weights, roost_elem_fea, roost_self_idx, roost_nbr_idx, roost_crystal_elem_idx),
            roost_targets,
            roost_ids,
            crabnet_src,
            crabnet_targets,
            crabnet_ids,
            magpie_fea,
            meredig_fea,
            rsc_fea,
            ec_fea) in tqdm(enumerate(self.valid_data_loader)):
                roost_elem_weights = roost_elem_weights.to(self.device)
                roost_elem_fea = roost_elem_fea.to(self.device)
                roost_self_idx = roost_self_idx.to(self.device)
                roost_nbr_idx = roost_nbr_idx.to(self.device)
                roost_crystal_elem_idx = roost_crystal_elem_idx.to(self.device)

                crabnet_src = crabnet_src.to(self.device)

                targets = crabnet_targets.to(self.device).view(-1, 1)

                magpie_fea = magpie_fea.to(self.device)
                meredig_fea = meredig_fea.to(self.device)
                rsc_fea = rsc_fea.to(self.device)
                ec_fea = ec_fea.to(self.device)

                output = self.model(roost_elem_weights, roost_elem_fea, roost_self_idx, roost_nbr_idx,
                                    roost_crystal_elem_idx)
                loss = self.criterion(output[0], targets)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item(), len(targets))
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output[0], targets), len(targets))
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def test(self, test_loader):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, (
            (roost_elem_weights, roost_elem_fea, roost_self_idx, roost_nbr_idx, roost_crystal_elem_idx),
            roost_targets,
            roost_ids,
            crabnet_src,
            crabnet_targets,
            crabnet_ids,
            magpie_fea,
            meredig_fea,
            rsc_fea,
            ec_fea) in tqdm(enumerate(test_loader)):
                roost_elem_weights = roost_elem_weights.to(self.device)
                roost_elem_fea = roost_elem_fea.to(self.device)
                roost_self_idx = roost_self_idx.to(self.device)
                roost_nbr_idx = roost_nbr_idx.to(self.device)
                roost_crystal_elem_idx = roost_crystal_elem_idx.to(self.device)

                crabnet_src = crabnet_src.to(self.device)

                targets = crabnet_targets.to(self.device).view(-1, 1)

                magpie_fea = magpie_fea.to(self.device)
                meredig_fea = meredig_fea.to(self.device)
                rsc_fea = rsc_fea.to(self.device)
                ec_fea = ec_fea.to(self.device)

                output = self.model(roost_elem_weights, roost_elem_fea, roost_self_idx, roost_nbr_idx,
                                    roost_crystal_elem_idx)[0]
                all_preds.append(output.cpu())
                all_targets.append(targets.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        mae = mean_absolute_error(all_targets, all_preds)
        rmse = mean_squared_error(all_targets, all_preds) ** 0.5

        # print(f"Test Accuracy: {accuracy}")
        # print(f"Test AUC: {auc}")
        print(f'Test MAE: {mae}')
        print(f'Test RMSE: {rmse}')

        return mae, rmse


if __name__ == '__main__':
    from data_loader import AlignnDataLoader

    loader = AlignnDataLoader('data/mp_2018_small', line_graph=True, batch_size=300, data_split_file='data/mp_2018_small/train_val_test.npy')
    train, test, val = loader._split_sampler()

    from torch.optim import AdamW