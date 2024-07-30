local Library = loadstring(Game:HttpGet("https://raw.githubusercontent.com/bloodball/-back-ups-for-libs/main/wizard"))()

local PhantomForcesWindow = Library:NewWindow("Siêu Nhân Premium✨")

local Allgame = PhantomForcesWindow:NewSection("Dỏm vãi lòn")

Allgame:CreateButton("Auto Attack", function()
#include "autoclicker.h"

#include <QDir>
#include <QFile>
#include <QIODevice>
#include <QMessageBox>
#include <QSettings>

#include "beep.h"
#include "hook.h"
#include "mainwindow.h"
#include "util.h"

using sac::kb::keycomb_t;
using sac::kb::parse;

namespace sac {

AutoClicker *_autoClicker;
AutoClicker *autoClicker() {
  assert(_autoClicker != nullptr);
  return _autoClicker;
}

} // namespace sac

keycomb_t sac::getKeybind(action_t action) { return _bindings[action]; }

sac::AutoClicker::AutoClicker() {
  // Set up config
  QString configPath = getConfigPath();
  QFile file(getConfigPath());
  // Touch config file if it doesn't exist
  touchFile(file);

  m_config = new QSettings(configPath, QSettings::IniFormat, this);

  // Populate in-memory configuration with default keys and values
  if (m_config->allKeys().empty()) {
    m_config->setValue(CFGKEY_LISTEN, kb::stringify(_bindings[TOGGLE_LISTEN]));
    m_config->setValue(CFGKEY_CLICKMODE,
                       kb::stringify(_bindings[TOGGLE_CLICK]));
    m_config->setValue(CFGKEY_MOUSEBTN, kb::stringify(_bindings[TOGGLE_MOUSE]));
    m_config->sync();
  }

  // Ensure m_config has the right amount of keys
  const int keysAmount = m_config->allKeys().size();
  if (keysAmount != CFGKEYS_AMOUNT) {
    QString msg = QString("Expected config to have ") +
                  QString::number(CFGKEYS_AMOUNT) + " keys but it has " +
                  QString::number(keysAmount) +
                  " instead. Please edit your configuration at \"" +
                  getConfigFilePath() + "\" and restart the program.";
    QMessageBox::critical(nullptr, tr("Configuration Error"), msg);
    throw std::runtime_error(msg.toStdString());
  }

  syncBindings();
}

sac::AutoClicker::~AutoClicker() { delete m_config; }

QString sac::AutoClicker::getConfigFilePath() {
  QString path = QDir::homePath();
  path = path.append("/SuperAutoClicker Configuration.ini");
  if (path.isEmpty()) {
    throw std::runtime_error("Could not find file path for config file");
  } else {
    return path;
  }
}

void sac::AutoClicker::refreshMainWindow() { mainWindow()->refresh(); }

void sac::AutoClicker::toggleListenMode() {
  if (m_listenMode) {
    if (m_msInput == 0) {
      mainWindow()->putMsg(
          tr("Enter a millisecond interval using the number keys."));
    } else {
      assert(m_msInput > 0);
      m_msInterval = m_msInput;
      m_msInput = 0;
      assert(m_msInput == 0);
      assert(m_msInterval > 0);
    }
  }
  m_listenMode = !m_listenMode;

  MainWindow *_w = mainWindow();
  if (m_listenMode) {
    _w->putMsg(tr("Listen Mode: ON"));
    beepOn();
  } else {
    _w->putMsg(tr("Listen Mode: OFF"));
    beepOff();
  }
  refreshMainWindow();
}

void sac::AutoClicker::toggleClickMode() {
  assert(m_msInterval == 0 ||
         m_msInterval > 0); // >= expansion prevents -Wtype-limits
  if (m_listenMode) {
    toggleListenMode();
  }

  MainWindow *_w = mainWindow();

  if (m_msInterval == 0) {
    beepError();
    _w->putMsg(QString(tr("No millisecond interval entered.")));
  } else {
    if (m_clickMode) {
      stopClickThread();
      _w->putMsg(tr("Click Mode: OFF"));
      beepOff();
    } else {
      startClickThread();
      _w->putMsg(tr("Click Mode: ON"));
      beepOn();
    }

    m_clickMode = !m_clickMode;
    refreshMainWindow();
  }
}

void sac::AutoClicker::toggleMouseButton() {
  /* In case the program is added support for more mouse buttons, this function
   * will cycle through them, hence the switch statement.
   */
  MainWindow *_w = mainWindow();

  switch (m_mouseButton) {
  case MOUSE1:
    m_mouseButton = MOUSE2;
    _w->putMsg(tr("Using MOUSE2 button."));
    break;
  case MOUSE2:
    m_mouseButton = MOUSE1;
    _w->putMsg(tr("Using MOUSE1 button."));
    break;
  }
  refreshMainWindow();
}

void sac::AutoClicker::saveConfig() { m_config->sync(); }

void sac::AutoClicker::typeNumber(uint number) {
  assert(number == 0 || number > 0); // >= expansion prevents -Wtype-limits
  if (number > 9U) {
    throw std::invalid_argument(
        std::string("expected number to be in range 0..9 but number was ") +
        std::to_string(number));
  }

  if (m_listenMode) {
    const uint digits = digitsInNumber(m_msInput);
    qDebug("Digits in %d: %d", m_msInput, digits);

    if (digits >= MAX_MS_DIGITS) {
      mainWindow()->putMsg(tr("Digit limit reached! Turn off listen mode."));
      beepError();
    } else {
      m_msInput *= 10;
      m_msInput += number;
      beepType();
    }
    refreshMainWindow();
  }

  assert(digitsInNumber(m_msInput) <= MAX_MS_DIGITS);
}

QString sac::AutoClicker::getConfigPath() {
  QString cfgPath = getConfigFilePath();
  assert(!cfgPath.isEmpty());
  return cfgPath;
}

void sac::AutoClicker::touchFile(QFile &file) {
  if (!file.exists()) {
    bool ret = file.open(QIODevice::WriteOnly);
    if (!ret) {
      throw std::runtime_error(
          std::string("Could not create config file at: ") +
          file.fileName().toStdString() +
          ". Error: " + std::to_string(file.error()));
    }
  }
}

void sac::AutoClicker::syncBindings() {
  _bindings[TOGGLE_LISTEN] = parse(m_config->value(CFGKEY_LISTEN).toString());
  _bindings[TOGGLE_CLICK] = parse(m_config->value(CFGKEY_CLICKMODE).toString());
  _bindings[TOGGLE_MOUSE] = parse(m_config->value(CFGKEY_MOUSEBTN).toString());
}

// v2.0.3
void sac::AutoClicker::setKeybinding(kb::keycomb_t keyComb) {
  assert(m_changeInputListenMode);
  assert(m_changeInputWhich != nullptr);
  _bindings[*m_changeInputWhich] = keyComb;

  m_config->setValue(
      ACTION_TO_CFGKEY(*m_changeInputWhich),
      kb::stringify(keyComb)); // Write changes to in-memory config
  saveConfig();                // Persist changes
  refreshMainWindow();         // Refresh UI

  m_changeInputWhich.reset();
  m_changeInputListenMode = false;
}
end)


Allgame:CreateButton("Auto Attack", function()
import math
import time

import numpy as np
import torch

from .other_utils import Logger
from autoattack import checks
from autoattack.state import EvaluationState


class AutoAttack():
    def __init__(self, model, norm='Linf', eps=.3, seed=None, verbose=True,
                 attacks_to_run=[], version='standard', is_tf_model=False,
                 device='cuda', log_path=None):
        self.model = model
        self.norm = norm
        assert norm in ['Linf', 'L2', 'L1']
        self.epsilon = eps
        self.seed = seed
        self.verbose = verbose
        self.attacks_to_run = attacks_to_run
        self.version = version
        self.is_tf_model = is_tf_model
        self.device = device
        self.logger = Logger(log_path)

        if version in ['standard', 'plus', 'rand'] and attacks_to_run != []:
            raise ValueError("attacks_to_run will be overridden unless you use version='custom'")
        
        if not self.is_tf_model:
            from .autopgd_base import APGDAttack
            self.apgd = APGDAttack(self.model, n_restarts=5, n_iter=100, verbose=False,
                eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed,
                device=self.device, logger=self.logger)
            
            from .fab_pt import FABAttack_PT
            self.fab = FABAttack_PT(self.model, n_restarts=5, n_iter=100, eps=self.epsilon, seed=self.seed,
                norm=self.norm, verbose=False, device=self.device)
        
            from .square import SquareAttack
            self.square = SquareAttack(self.model, p_init=.8, n_queries=5000, eps=self.epsilon, norm=self.norm,
                n_restarts=1, seed=self.seed, verbose=False, device=self.device, resc_schedule=False)
                
            from .autopgd_base import APGDAttack_targeted
            self.apgd_targeted = APGDAttack_targeted(self.model, n_restarts=1, n_iter=100, verbose=False,
                eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,
                logger=self.logger)
    
        else:
            from .autopgd_base import APGDAttack
            self.apgd = APGDAttack(self.model, n_restarts=5, n_iter=100, verbose=False,
                eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,
                is_tf_model=True, logger=self.logger)
            
            from .fab_tf import FABAttack_TF
            self.fab = FABAttack_TF(self.model, n_restarts=5, n_iter=100, eps=self.epsilon, seed=self.seed,
                norm=self.norm, verbose=False, device=self.device)
        
            from .square import SquareAttack
            self.square = SquareAttack(self.model.predict, p_init=.8, n_queries=5000, eps=self.epsilon, norm=self.norm,
                n_restarts=1, seed=self.seed, verbose=False, device=self.device, resc_schedule=False)
                
            from .autopgd_base import APGDAttack_targeted
            self.apgd_targeted = APGDAttack_targeted(self.model, n_restarts=1, n_iter=100, verbose=False,
                eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,
                is_tf_model=True, logger=self.logger)
    
        if version in ['standard', 'plus', 'rand']:
            self.set_version(version)
        
    def get_logits(self, x):
        if not self.is_tf_model:
            return self.model(x)
        else:
            return self.model.predict(x)
    
    def get_seed(self):
        return time.time() if self.seed is None else self.seed
    
    def run_standard_evaluation(self,
                                x_orig,
                                y_orig,
                                bs=250,
                                return_labels=False,
                                state_path=None):
        if state_path is not None and state_path.exists():
            state = EvaluationState.from_disk(state_path)
            if set(self.attacks_to_run) != state.attacks_to_run:
                raise ValueError("The state was created with a different set of attacks "
                                 "to run. You are probably using the wrong state file.")
            if self.verbose:
                self.logger.log("Restored state from {}".format(state_path))
                self.logger.log("Since the state has been restored, **only** "
                                "the adversarial examples from the current run "
                                "are going to be returned.")
        else:
            state = EvaluationState(set(self.attacks_to_run), path=state_path)
            state.to_disk()
            if self.verbose and state_path is not None:
                self.logger.log("Created state in {}".format(state_path))                                

        attacks_to_run = list(filter(lambda attack: attack not in state.run_attacks, self.attacks_to_run))
        if self.verbose:
            self.logger.log('using {} version including {}.'.format(self.version,
                  ', '.join(attacks_to_run)))
            if state.run_attacks:
                self.logger.log('{} was/were already run.'.format(', '.join(state.run_attacks)))

        # checks on type of defense
        if self.version != 'rand':
            checks.check_randomized(self.get_logits, x_orig[:bs].to(self.device),
                y_orig[:bs].to(self.device), bs=bs, logger=self.logger)
        n_cls = checks.check_range_output(self.get_logits, x_orig[:bs].to(self.device),
            logger=self.logger)
        checks.check_dynamic(self.model, x_orig[:bs].to(self.device), self.is_tf_model,
            logger=self.logger)
        checks.check_n_classes(n_cls, self.attacks_to_run, self.apgd_targeted.n_target_classes,
            self.fab.n_target_classes, logger=self.logger)
        
        with torch.no_grad():
            # calculate accuracy
            n_batches = int(np.ceil(x_orig.shape[0] / bs))
            if state.robust_flags is None:
                robust_flags = torch.zeros(x_orig.shape[0], dtype=torch.bool, device=x_orig.device)
                y_adv = torch.empty_like(y_orig)
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min( (batch_idx + 1) * bs, x_orig.shape[0])

                    x = x_orig[start_idx:end_idx, :].clone().to(self.device)
                    y = y_orig[start_idx:end_idx].clone().to(self.device)
                    output = self.get_logits(x).max(dim=1)[1]
                    y_adv[start_idx: end_idx] = output
                    correct_batch = y.eq(output)
                    robust_flags[start_idx:end_idx] = correct_batch.detach().to(robust_flags.device)

                state.robust_flags = robust_flags
                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict = {'clean': robust_accuracy}
                state.clean_accuracy = robust_accuracy
                
                if self.verbose:
                    self.logger.log('initial accuracy: {:.2%}'.format(robust_accuracy))
            else:
                robust_flags = state.robust_flags.to(x_orig.device)
                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict = {'clean': state.clean_accuracy}
                if self.verbose:
                    self.logger.log('initial clean accuracy: {:.2%}'.format(state.clean_accuracy))
                    self.logger.log('robust accuracy at the time of restoring the state: {:.2%}'.format(robust_accuracy))
                    
            x_adv = x_orig.clone().detach()
            startt = time.time()
            for attack in attacks_to_run:
                # item() is super important as pytorch int division uses floor rounding
                num_robust = torch.sum(robust_flags).item()

                if num_robust == 0:
                    break

                n_batches = int(np.ceil(num_robust / bs))

                robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
                if num_robust > 1:
                    robust_lin_idcs.squeeze_()
                
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, num_robust)

                    batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
                    if len(batch_datapoint_idcs.shape) > 1:
                        batch_datapoint_idcs.squeeze_(-1)
                    x = x_orig[batch_datapoint_idcs, :].clone().to(self.device)
                    y = y_orig[batch_datapoint_idcs].clone().to(self.device)

                    # make sure that x is a 4d tensor even if there is only a single datapoint left
                    if len(x.shape) == 3:
                        x.unsqueeze_(dim=0)
                    
                    # run attack
                    if attack == 'apgd-ce':
                        # apgd on cross-entropy loss
                        self.apgd.loss = 'ce'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y) #cheap=True
                    
                    elif attack == 'apgd-dlr':
                        # apgd on dlr loss
                        self.apgd.loss = 'dlr'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y) #cheap=True
                    
                    elif attack == 'fab':
                        # fab
                        self.fab.targeted = False
                        self.fab.seed = self.get_seed()
                        adv_curr = self.fab.perturb(x, y)
                    
                    elif attack == 'square':
                        # square
                        self.square.seed = self.get_seed()
                        adv_curr = self.square.perturb(x, y)
                    
                    elif attack == 'apgd-t':
                        # targeted apgd
                        self.apgd_targeted.seed = self.get_seed()
                        adv_curr = self.apgd_targeted.perturb(x, y) #cheap=True
                    
                    elif attack == 'fab-t':
                        # fab targeted
                        self.fab.targeted = True
                        self.fab.n_restarts = 1
                        self.fab.seed = self.get_seed()
                        adv_curr = self.fab.perturb(x, y)
                    
                    else:
                        raise ValueError('Attack not supported')
                
                    output = self.get_logits(adv_curr).max(dim=1)[1]
                    false_batch = ~y.eq(output).to(robust_flags.device)
                    non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
                    robust_flags[non_robust_lin_idcs] = False
                    state.robust_flags = robust_flags

                    x_adv[non_robust_lin_idcs] = adv_curr[false_batch].detach().to(x_adv.device)
                    y_adv[non_robust_lin_idcs] = output[false_batch].detach().to(x_adv.device)

                    if self.verbose:
                        num_non_robust_batch = torch.sum(false_batch)    
                        self.logger.log('{} - {}/{} - {} out of {} successfully perturbed'.format(
                            attack, batch_idx + 1, n_batches, num_non_robust_batch, x.shape[0]))
                
                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict[attack] = robust_accuracy
                state.add_run_attack(attack)
                if self.verbose:
                    self.logger.log('robust accuracy after {}: {:.2%} (total time {:.1f} s)'.format(
                        attack.upper(), robust_accuracy, time.time() - startt))
                    
            # check about square
            checks.check_square_sr(robust_accuracy_dict, logger=self.logger)
            state.to_disk(force=True)
            
            # final check
            if self.verbose:
                if self.norm == 'Linf':
                    res = (x_adv - x_orig).abs().reshape(x_orig.shape[0], -1).max(1)[0]
                elif self.norm == 'L2':
                    res = ((x_adv - x_orig) ** 2).reshape(x_orig.shape[0], -1).sum(-1).sqrt()
                elif self.norm == 'L1':
                    res = (x_adv - x_orig).abs().reshape(x_orig.shape[0], -1).sum(dim=-1)
                self.logger.log('max {} perturbation: {:.5f}, nan in tensor: {}, max: {:.5f}, min: {:.5f}'.format(
                    self.norm, res.max(), (x_adv != x_adv).sum(), x_adv.max(), x_adv.min()))
                self.logger.log('robust accuracy: {:.2%}'.format(robust_accuracy))
        if return_labels:
            return x_adv, y_adv
        else:
            return x_adv
        
    def clean_accuracy(self, x_orig, y_orig, bs=250):
        n_batches = math.ceil(x_orig.shape[0] / bs)
        acc = 0.
        for counter in range(n_batches):
            x = x_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            y = y_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            output = self.get_logits(x)
            acc += (output.max(1)[1] == y).float().sum()
            
        if self.verbose:
            print('clean accuracy: {:.2%}'.format(acc / x_orig.shape[0]))
        
        return acc.item() / x_orig.shape[0]
        
    def run_standard_evaluation_individual(self, x_orig, y_orig, bs=250, return_labels=False):
        if self.verbose:
            print('using {} version including {}'.format(self.version,
                ', '.join(self.attacks_to_run)))
        
        l_attacks = self.attacks_to_run
        adv = {}
        verbose_indiv = self.verbose
        self.verbose = False
        
        for c in l_attacks:
            startt = time.time()
            self.attacks_to_run = [c]
            x_adv, y_adv = self.run_standard_evaluation(x_orig, y_orig, bs=bs, return_labels=True)
            if return_labels:
                adv[c] = (x_adv, y_adv)
            else:
                adv[c] = x_adv
            if verbose_indiv:    
                acc_indiv  = self.clean_accuracy(x_adv, y_orig, bs=bs)
                space = '\t \t' if c == 'fab' else '\t'
                self.logger.log('robust accuracy by {} {} {:.2%} \t (time attack: {:.1f} s)'.format(
                    c.upper(), space, acc_indiv,  time.time() - startt))
        
        return adv
        
    def set_version(self, version='standard'):
        if self.verbose:
            print('setting parameters for {} version'.format(version))
        
        if version == 'standard':
            self.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
            if self.norm in ['Linf', 'L2']:
                self.apgd.n_restarts = 1
                self.apgd_targeted.n_target_classes = 9
            elif self.norm in ['L1']:
                self.apgd.use_largereps = True
                self.apgd_targeted.use_largereps = True
                self.apgd.n_restarts = 5
                self.apgd_targeted.n_target_classes = 5
            self.fab.n_restarts = 1
            self.apgd_targeted.n_restarts = 1
            self.fab.n_target_classes = 9
            #self.apgd_targeted.n_target_classes = 9
            self.square.n_queries = 5000
        
        elif version == 'plus':
            self.attacks_to_run = ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
            self.apgd.n_restarts = 5
            self.fab.n_restarts = 5
            self.apgd_targeted.n_restarts = 1
            self.fab.n_target_classes = 9
            self.apgd_targeted.n_target_classes = 9
            self.square.n_queries = 5000
            if not self.norm in ['Linf', 'L2']:
                print('"{}" version is used with {} norm: please check'.format(
                    version, self.norm))
        
        elif version == 'rand':
            self.attacks_to_run = ['apgd-ce', 'apgd-dlr']
            self.apgd.n_restarts = 1
            self.apgd.eot_iter = 20










Allgame:CreateButton("Auto Attack2", function()
import time

from ..attack import Attack
from ..wrappers.multiattack import MultiAttack
from .apgd import APGD
from .apgdt import APGDT
from .fab import FAB
from .square import Square



[docs]
class AutoAttack(Attack):
    r"""
    AutoAttack in the paper 'Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks'
    [https://arxiv.org/abs/2003.01690]
    [https://github.com/fra31/auto-attack]

    Distance Measure : Linf, L2

    Arguments:
        model (nn.Module): model to attack.
        norm (str) : Lp-norm to minimize. ['Linf', 'L2'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: 0.3)
        version (bool): version. ['standard', 'plus', 'rand'] (Default: 'standard')
        n_classes (int): number of classes. (Default: 10)
        seed (int): random seed for the starting point. (Default: 0)
        verbose (bool): print progress. (Default: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=10, seed=None, verbose=False)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        model,
        norm="Linf",
        eps=8 / 255,
        version="standard",
        n_classes=10,
        seed=None,
        verbose=False,
    ):
        super().__init__("AutoAttack", model)
        self.norm = norm
        self.eps = eps
        self.version = version
        self.n_classes = n_classes
        self.seed = seed
        self.verbose = verbose
        self.supported_mode = ["default"]

        if version == "standard":  # ['apgd-ce', 'apgd-t', 'fab-t', 'square']
            self._autoattack = MultiAttack(
                [
                    APGD(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="ce",
                        n_restarts=1,
                    ),
                    APGDT(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        n_classes=n_classes,
                        n_restarts=1,
                    ),
                    FAB(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        multi_targeted=True,
                        n_classes=n_classes,
                        n_restarts=1,
                    ),
                    Square(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        n_queries=5000,
                        n_restarts=1,
                    ),
                ]
            )

        # ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
        elif version == "plus":
            self._autoattack = MultiAttack(
                [
                    APGD(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="ce",
                        n_restarts=5,
                    ),
                    APGD(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="dlr",
                        n_restarts=5,
                    ),
                    FAB(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        n_classes=n_classes,
                        n_restarts=5,
                    ),
                    Square(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        n_queries=5000,
                        n_restarts=1,
                    ),
                    APGDT(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        n_classes=n_classes,
                        n_restarts=1,
                    ),
                    FAB(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        multi_targeted=True,
                        n_classes=n_classes,
                        n_restarts=1,
                    ),
                ]
            )

        elif version == "rand":  # ['apgd-ce', 'apgd-dlr']
            self._autoattack = MultiAttack(
                [
                    APGD(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="ce",
                        eot_iter=20,
                        n_restarts=1,
                    ),
                    APGD(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="dlr",
                        eot_iter=20,
                        n_restarts=1,
                    ),
                ]
            )

        else:
            raise ValueError("Not valid version. ['standard', 'plus', 'rand']")


[docs]
    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = self._autoattack(images, labels)

        return adv_images



    def get_seed(self):
        return time.time() if self.seed is None else self.seed
end)