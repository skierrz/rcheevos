#include "rc_internal.h"

#include <stddef.h>
#include <string.h> /* memset */

void rc_parse_trigger_internal(rc_trigger_t* self, const char** memaddr, rc_parse_state_t* parse) {
  rc_condset_t** next;
  const char* aux;

  aux = *memaddr;
  next = &self->alternative;

  /* reset in case multiple triggers are parsed by the same parse_state */
  parse->measured_target = 0;
  parse->has_required_hits = 0;
  parse->measured_as_percent = 0;

  if (*aux == 's' || *aux == 'S') {
    self->requirement = NULL;
  }
  else {
    self->requirement = rc_parse_condset(&aux, parse);

    if (parse->offset < 0)
      return;

    self->requirement->next = NULL;
  }

  while (*aux == 's' || *aux == 'S') {
    aux++;
    *next = rc_parse_condset(&aux, parse);

    if (parse->offset < 0) {
      return;
    }

    next = &(*next)->next;
  }

  *next = NULL;
  *memaddr = aux;

  self->measured_target = parse->measured_target;
  self->measured_value = parse->measured_target ? RC_MEASURED_UNKNOWN : 0;
  self->measured_as_percent = parse->measured_as_percent;
  self->state = RC_TRIGGER_STATE_WAITING;
  self->has_hits = 0;
  self->has_memrefs = 0;
  self->cd_flags = 0;
  self->cd_state = 0;
  self->cd_frame = 0;
}

int rc_trigger_size(const char* memaddr) {
  rc_trigger_with_memrefs_t* trigger;
  rc_preparse_state_t preparse;
  rc_init_preparse_state(&preparse);

  trigger = RC_ALLOC(rc_trigger_with_memrefs_t, &preparse.parse);
  rc_parse_trigger_internal(&trigger->trigger, &memaddr, &preparse.parse);
  rc_preparse_alloc_memrefs(NULL, &preparse);

  rc_destroy_preparse_state(&preparse);
  return preparse.parse.offset;
}

rc_trigger_t* rc_parse_trigger(void* buffer, const char* memaddr, void* unused_L, int unused_funcs_idx) {
  rc_trigger_with_memrefs_t* trigger;
  rc_preparse_state_t preparse;
  const char* preparse_memaddr = memaddr;

  (void)unused_L;
  (void)unused_funcs_idx;

  if (!buffer || !memaddr)
    return NULL;

  /* first pass : determine how many memrefs are needed */
  rc_init_preparse_state(&preparse);
  trigger = RC_ALLOC(rc_trigger_with_memrefs_t, &preparse.parse);
  rc_parse_trigger_internal(&trigger->trigger, &preparse_memaddr, &preparse.parse);

  /* allocate the trigger and memrefs */
  rc_reset_parse_state(&preparse.parse, buffer);
  trigger = RC_ALLOC(rc_trigger_with_memrefs_t, &preparse.parse);
  rc_preparse_alloc_memrefs(&trigger->memrefs, &preparse);

  /* parse the trigger */
  rc_parse_trigger_internal(&trigger->trigger, &memaddr, &preparse.parse);
  trigger->trigger.has_memrefs = 1;

  rc_destroy_preparse_state(&preparse);
  return (preparse.parse.offset >= 0) ? &trigger->trigger : NULL;
}

int rc_trigger_state_active(int state)
{
  switch (state)
  {
    case RC_TRIGGER_STATE_DISABLED:
    case RC_TRIGGER_STATE_INACTIVE:
    case RC_TRIGGER_STATE_TRIGGERED:
      return 0;

    default:
      return 1;
  }
}

static int rc_condset_is_measured_from_hitcount(const rc_condset_t* condset, uint32_t measured_value)
{
  const rc_condition_t* condition;
  for (condition = condset->conditions; condition; condition = condition->next) {
    if (condition->type == RC_CONDITION_MEASURED && condition->required_hits &&
        condition->current_hits == measured_value) {
      return 1;
    }
  }

  return 0;
}

static void rc_reset_trigger_hitcounts(rc_trigger_t* self) {
  rc_condset_t* condset;

  if (self->requirement) {
    rc_reset_condset(self->requirement);
  }

  condset = self->alternative;

  while (condset) {
    rc_reset_condset(condset);
    condset = condset->next;
  }
}

static void rc_update_trigger_memrefs(rc_trigger_t* self, rc_read_memory_func_t read_memory, void* ud) {
  if (self->has_memrefs) {
    rc_trigger_with_memrefs_t* trigger = (rc_trigger_with_memrefs_t*)self;
    rc_update_memref_values(&trigger->memrefs, read_memory, ud);
  }
}

rc_memrefs_t* rc_trigger_get_memrefs(rc_trigger_t* self) {
  if (self->has_memrefs) {
    rc_trigger_with_memrefs_t* trigger = (rc_trigger_with_memrefs_t*)self;
    return &trigger->memrefs;
  }

  return NULL;
}

int rc_evaluate_trigger(rc_trigger_t* self, rc_read_memory_func_t read_memory, void* ud, void* unused_L) {
  rc_eval_state_t eval_state;
  rc_condset_t* condset;
  rc_typed_value_t measured_value;
  int measured_from_hits = 0;
  int ret;
  char is_paused;
  char is_primed;

  (void)unused_L;

  switch (self->state)
  {
    case RC_TRIGGER_STATE_TRIGGERED:
      /* previously triggered. do nothing - return INACTIVE so caller doesn't think it triggered again */
      return RC_TRIGGER_STATE_INACTIVE;

    case RC_TRIGGER_STATE_DISABLED:
      /* unsupported. do nothing - return INACTIVE */
      return RC_TRIGGER_STATE_INACTIVE;

    case RC_TRIGGER_STATE_INACTIVE:
      /* not yet active. update the memrefs so deltas are correct when it becomes active, then return INACTIVE */
      rc_update_trigger_memrefs(self, read_memory, ud);
      return RC_TRIGGER_STATE_INACTIVE;

    default:
      break;
  }

  /* update the memory references */
  rc_update_trigger_memrefs(self, read_memory, ud);

  /* process the trigger */
  memset(&eval_state, 0, sizeof(eval_state));
  eval_state.read_memory = read_memory;
  eval_state.read_memory_userdata = ud;

  measured_value.type = RC_VALUE_TYPE_NONE;

  if (self->requirement != NULL) {
    ret = rc_test_condset(self->requirement, &eval_state);
    is_paused = eval_state.is_paused;
    is_primed = eval_state.is_primed;

    if (eval_state.measured_value.type != RC_VALUE_TYPE_NONE) {
      memcpy(&measured_value, &eval_state.measured_value, sizeof(measured_value));
      measured_from_hits = eval_state.measured_from_hits;
    }
  } else {
    ret = 1;
    is_paused = 0;
    is_primed = 1;
  }

  condset = self->alternative;
  if (condset) {
    int sub = 0;
    char sub_paused = 1;
    char sub_primed = 0;

    do {
      sub |= rc_test_condset(condset, &eval_state);
      sub_paused &= eval_state.is_paused;
      sub_primed |= eval_state.is_primed;

      if (eval_state.measured_value.type != RC_VALUE_TYPE_NONE) {
        /* if no previous Measured value was captured, or the new one is greater, keep the new one */
        if (measured_value.type == RC_VALUE_TYPE_NONE ||
            rc_typed_value_compare(&eval_state.measured_value, &measured_value, RC_OPERATOR_GT)) {
          memcpy(&measured_value, &eval_state.measured_value, sizeof(measured_value));
          measured_from_hits = eval_state.measured_from_hits;
        }
      }

      condset = condset->next;
    } while (condset);

    /* to trigger, the core must be true and at least one alt must be true */
    ret &= sub;
    is_primed &= sub_primed;

    /* if the core is not paused, all alts must be paused to count as a paused trigger */
    is_paused |= sub_paused;
  }

  if (is_paused) {
    /* if the trigger is fully paused, ignore any updates to the measured value */
  }
  else if (measured_value.type == RC_VALUE_TYPE_NONE) {
    /* if a measured value was not captured, keep the old value (it's possible to pause
     * an alt that is generating the measured value without fully pausing the trigger) */
  }
  else {
    rc_typed_value_convert(&measured_value, RC_VALUE_TYPE_UNSIGNED);
    self->measured_value = measured_value.value.u32;
  }

  /* if any ResetIf condition was true, reset the hit counts */
  if (eval_state.was_reset) {
    /* if the measured value came from a hit count, reset it. do this before calling
     * rc_reset_trigger_hitcounts in case we need to call rc_condset_is_measured_from_hitcount */
    if (measured_from_hits) {
      self->measured_value = 0;
    }
    else if (is_paused && self->measured_value) {
      /* if the measured value is in a paused group, measured_from_hits won't have been set.
       * attempt to determine if it should have been */
      if (self->requirement && self->requirement->is_paused &&
          rc_condset_is_measured_from_hitcount(self->requirement, self->measured_value)) {
        self->measured_value = 0;
      }
      else {
        for (condset = self->alternative; condset; condset = condset->next) {
          if (condset->is_paused && rc_condset_is_measured_from_hitcount(condset, self->measured_value)) {
            self->measured_value = 0;
            break;
          }
        }
      }
    }

    rc_reset_trigger_hitcounts(self);

    /* if there were hit counts to clear, return RESET, but don't change the state */
    if (self->has_hits) {
      self->has_hits = 0;

      /* cannot be PRIMED while ResetIf is true */
      if (self->state == RC_TRIGGER_STATE_PRIMED)
        self->state = RC_TRIGGER_STATE_ACTIVE;

      return RC_TRIGGER_STATE_RESET;
    }

    /* any hits that were tallied were just reset */
    eval_state.has_hits = 0;
    is_primed = 0;
  }
  else if (ret) {
    /* if the state is WAITING and the trigger is ready to fire, ignore it and reset the hit counts */
    if (self->state == RC_TRIGGER_STATE_WAITING) {
      rc_reset_trigger(self);
      self->has_hits = 0;
      return RC_TRIGGER_STATE_WAITING;
    }

    /* trigger was triggered */
    self->state = RC_TRIGGER_STATE_TRIGGERED;
    return RC_TRIGGER_STATE_TRIGGERED;
  }

  /* did not trigger this frame - update the information we'll need for next time */
  self->has_hits = eval_state.has_hits;

  if (is_paused) {
    self->state = RC_TRIGGER_STATE_PAUSED;
  }
  else if (is_primed) {
    self->state = RC_TRIGGER_STATE_PRIMED;
  }
  else {
    self->state = RC_TRIGGER_STATE_ACTIVE;
  }

  /* if an individual condition was reset, notify the caller */
  if (eval_state.was_cond_reset)
    return RC_TRIGGER_STATE_RESET;

  /* otherwise, just return the current state */
  return self->state;
}

int rc_test_trigger(rc_trigger_t* self, rc_read_memory_func_t read_memory, void* ud, void* unused_L) {
  /* for backwards compatibilty, rc_test_trigger always assumes the achievement is active */
  self->state = RC_TRIGGER_STATE_ACTIVE;

  return (rc_evaluate_trigger(self, read_memory, ud, unused_L) == RC_TRIGGER_STATE_TRIGGERED);
}

void rc_reset_trigger(rc_trigger_t* self) {
  if (!self)
    return;

  rc_reset_trigger_hitcounts(self);

  self->state = RC_TRIGGER_STATE_WAITING;

  if (self->measured_target)
    self->measured_value = RC_MEASURED_UNKNOWN;

  self->has_hits = 0;
  self->cd_flags = 0;
}

/* Change-driven evaluation: skip a trigger whose evaluation can't change.
 *
 * Evaluating a trigger is a function of its state (hit counts, is_true,
 * is_paused, state, has_hits, measured_value) and its operand values; for
 * rc_client's triggers nothing else (they have no memrefs of their own, and
 * Remember/Recall resolve to memrefs). An operand's value, delta and prior
 * are all unchanged at frame f if its memref's changed flag was clear at f
 * and at f-1. So if every memref the trigger reads was unchanged at f and
 * f-1, and the evaluation at f-1 changed none of its state and returned
 * trigger->state, the evaluation at f would do exactly the same: skipping it
 * and returning trigger->state gives the same result, the same state and the
 * same bytes. A skipped frame keeps the trigger clean, so it stays skipped
 * until a memref it reads changes.
 *
 * frame: the caller's count of memref updates. A trigger not evaluated on
 * the update before (cd_frame != frame - 1) can't be skipped: its memrefs
 * may have changed in between unseen. Any other change to a trigger's state
 * goes through rc_reset_trigger or the progress reader (they clear
 * cd_flags) or changes trigger->state (checked against cd_state). */
#define RC_CD_CLEAN 1       /* the last evaluation changed no state and returned trigger->state */
#define RC_CD_DEP_PREV 2    /* a memref it reads changed at the last evaluation */
#define RC_CD_MAX_CONDITIONS 128

static int rc_operand_changed(const rc_operand_t* operand) {
  switch (operand->type) {
    case RC_OPERAND_CONST:
    case RC_OPERAND_FP:
    case RC_OPERAND_FUNC:
      return 0;

    case RC_OPERAND_RECALL:
      if (!rc_operand_type_is_memref(operand->memref_access_type) || !operand->value.memref)
        return 0;
      return operand->value.memref->value.changed;

    default:
      return operand->value.memref->value.changed;
  }
}

static int rc_condset_memrefs_changed(const rc_condset_t* condset) {
  const rc_condition_t* condition;
  for (condition = condset->conditions; condition; condition = condition->next) {
    if (rc_operand_changed(&condition->operand1) || rc_operand_changed(&condition->operand2))
      return 1;
  }
  return 0;
}

static int rc_trigger_memrefs_changed(const rc_trigger_t* self) {
  const rc_condset_t* condset;
  if (self->requirement && rc_condset_memrefs_changed(self->requirement))
    return 1;
  for (condset = self->alternative; condset; condset = condset->next) {
    if (rc_condset_memrefs_changed(condset))
      return 1;
  }
  return 0;
}

/* the state an evaluation can change, for one comparison before and after */
typedef struct rc_trigger_snapshot_t {
  uint32_t hits[RC_CD_MAX_CONDITIONS];
  uint8_t flags[RC_CD_MAX_CONDITIONS];   /* is_true; is_paused on a condset's first entry */
  uint32_t count;
  uint32_t measured_value;
  uint8_t state;
  uint8_t has_hits;
} rc_trigger_snapshot_t;

static int rc_condset_snapshot(const rc_condset_t* condset, rc_trigger_snapshot_t* snapshot, int compare) {
  const rc_condition_t* condition;
  uint32_t i = snapshot->count;
  if (!condset->conditions)   /* nowhere to keep is_paused: never clean */
    return 0;
  for (condition = condset->conditions; condition; condition = condition->next, ++i) {
    uint8_t flags = (uint8_t)(condition->is_true | (condition == condset->conditions ? (condset->is_paused ? 0x80 : 0) : 0));
    if (i >= RC_CD_MAX_CONDITIONS)
      return 0;
    if (compare) {
      if (snapshot->hits[i] != condition->current_hits || snapshot->flags[i] != flags)
        return 0;
    }
    else {
      snapshot->hits[i] = condition->current_hits;
      snapshot->flags[i] = flags;
    }
  }
  snapshot->count = i;
  return 1;
}

/* compare 0: take the snapshot (returns 0 if the trigger is too big); compare 1: returns 1 if unchanged */
static int rc_trigger_snapshot(const rc_trigger_t* self, rc_trigger_snapshot_t* snapshot, int compare) {
  const rc_condset_t* condset;
  snapshot->count = 0;
  if (compare) {
    if (snapshot->measured_value != self->measured_value || snapshot->state != self->state || snapshot->has_hits != self->has_hits)
      return 0;
  }
  else {
    snapshot->measured_value = self->measured_value;
    snapshot->state = self->state;
    snapshot->has_hits = self->has_hits;
  }
  if (self->requirement && !rc_condset_snapshot(self->requirement, snapshot, compare))
    return 0;
  for (condset = self->alternative; condset; condset = condset->next) {
    if (!rc_condset_snapshot(condset, snapshot, compare))
      return 0;
  }
  return 1;
}

int rc_evaluate_trigger_change_driven(rc_trigger_t* self, rc_read_memory_func_t read_memory, void* ud, uint32_t frame) {
  rc_trigger_snapshot_t snapshot;
  int changed, followed, have_snapshot, ret;

  if (self->has_memrefs || !rc_trigger_state_active(self->state)) {
    /* not a candidate: its own memrefs, or a state rc_evaluate_trigger handles without evaluating */
    self->cd_flags = 0;
    return rc_evaluate_trigger(self, read_memory, ud, NULL);
  }

  changed = rc_trigger_memrefs_changed(self);
  followed = (self->cd_frame == frame - 1);   /* evaluated on the update before this one */

  if (!changed && followed && (self->cd_flags & (RC_CD_CLEAN | RC_CD_DEP_PREV)) == RC_CD_CLEAN &&
      self->state == self->cd_state) {
    self->cd_frame = frame;
    return self->state;
  }

  /* the evaluation can only make the trigger skippable next frame if its memrefs are unchanged now */
  have_snapshot = !changed && rc_trigger_snapshot(self, &snapshot, 0);

  ret = rc_evaluate_trigger(self, read_memory, ud, NULL);

  self->cd_flags = changed ? RC_CD_DEP_PREV : 0;
  if (have_snapshot && ret == self->state && ret == snapshot.state && rc_trigger_snapshot(self, &snapshot, 1))
    self->cd_flags |= RC_CD_CLEAN;
  self->cd_state = self->state;
  self->cd_frame = frame;
  return ret;
}
