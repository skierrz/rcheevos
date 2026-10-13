#include "rc_internal.h"

#include <stdlib.h> /* malloc/realloc */
#include <string.h> /* memcpy */
#include <math.h>   /* INFINITY/NAN */

#define MEMREF_PLACEHOLDER_ADDRESS 0xFFFFFFFF

union rc_bigendian_check_t {
    uint32_t u32;
    uint8_t u8[4];
};
static const union rc_bigendian_check_t host_bigendian_check = { 0x01000000 };
static uint8_t rc_is_host_bigendian() { return host_bigendian_check.u8[0]; }

/* memref deduplication looks up each new memref in every existing one. for lists created by
 * rc_memrefs_init (a runtime's shared list, which can hold thousands of entries), a hash index
 * replaces the linear scan. the hash covers only fields that the matching logic compares, so
 * entries that match always share a hash; and with linear probing and no removals, entries with
 * the same hash are probed in the order they were added (list order), so a lookup returns the
 * same entry the linear scan would. */
/* how deep the hash follows modified memrefs. deeper only separates more pointer chains. */
#define RC_MEMREF_HASH_LEVELS 2

static uint32_t rc_memref_hash_mix(uint32_t hash, uint32_t value) {
  return (hash ^ value) * 0x9E3779B1;
}

static uint32_t rc_memref_hash_modifiers(uint32_t hash, const rc_operand_t* parent,
                                         uint8_t modifier_type, const rc_operand_t* modifier, int levels);

/* hashes what rc_operands_are_equal compares (but no floating point values, as 0.0 == -0.0),
 * following modified memrefs at most levels deep */
static uint32_t rc_memref_hash_operand(const rc_operand_t* operand, int levels) {
  uint32_t hash = operand->type;
  const rc_memref_t* memref;

  switch (operand->type) {
    case RC_OPERAND_CONST:
      return rc_memref_hash_mix(hash, operand->value.num);
    case RC_OPERAND_FP:
      return hash;
    case RC_OPERAND_RECALL:
      hash = rc_memref_hash_mix(hash, operand->memref_access_type);
      if (operand->memref_access_type == RC_OPERAND_CONST)
        return rc_memref_hash_mix(hash, operand->value.num);
      if (operand->memref_access_type == RC_OPERAND_FP)
        return hash;
      break;
    default:
      break;
  }

  hash = rc_memref_hash_mix(hash, operand->size);
  memref = operand->value.memref;
  if (!memref)
    return hash;

  hash = rc_memref_hash_mix(hash, memref->value.memref_type);
  switch (memref->value.memref_type) {
    case RC_MEMREF_TYPE_MEMREF:
      hash = rc_memref_hash_mix(hash, memref->address);
      return rc_memref_hash_mix(hash, memref->value.size);
    case RC_MEMREF_TYPE_MODIFIED_MEMREF: {
      const rc_modified_memref_t* modified_memref = (const rc_modified_memref_t*)memref;
      if (levels == 0)
        return hash;
      return rc_memref_hash_modifiers(hash, &modified_memref->parent, modified_memref->modifier_type,
                                      &modified_memref->modifier, levels - 1);
    }
    default:
      return hash;
  }
}

static uint32_t rc_memref_hash_modifiers(uint32_t hash, const rc_operand_t* parent,
                                         uint8_t modifier_type, const rc_operand_t* modifier, int levels) {
  hash = rc_memref_hash_mix(hash, modifier_type);
  hash = rc_memref_hash_mix(hash, rc_memref_hash_operand(parent, levels));
  return rc_memref_hash_mix(hash, rc_memref_hash_operand(modifier, levels));
}

static uint32_t rc_memref_hash(uint32_t address, uint8_t size) {
  const uint32_t hash = rc_memref_hash_mix(rc_memref_hash_mix(RC_MEMREF_TYPE_MEMREF, address), size);
  return hash ^ (hash >> 16);
}

static uint32_t rc_modified_memref_hash(uint8_t size, const rc_operand_t* parent,
                                        uint8_t modifier_type, const rc_operand_t* modifier) {
  const uint32_t hash = rc_memref_hash_modifiers(rc_memref_hash_mix(RC_MEMREF_TYPE_MODIFIED_MEMREF, size),
                                                 parent, modifier_type, modifier, RC_MEMREF_HASH_LEVELS);
  return hash ^ (hash >> 16);
}

static void rc_memrefs_index_insert(rc_memrefs_t* memrefs, rc_memref_t* memref) {
  const uint32_t mask = memrefs->index_capacity - 1;
  uint32_t slot;

  if (memref->value.memref_type == RC_MEMREF_TYPE_MODIFIED_MEMREF) {
    const rc_modified_memref_t* modified_memref = (const rc_modified_memref_t*)memref;
    slot = rc_modified_memref_hash(memref->value.size, &modified_memref->parent,
                                   modified_memref->modifier_type, &modified_memref->modifier);
  }
  else {
    slot = rc_memref_hash(memref->address, memref->value.size);
  }

  for (slot &= mask; memrefs->index[slot]; slot = (slot + 1) & mask)
    ;

  memrefs->index[slot] = memref;
  memrefs->index_count++;
}

/* (re)builds the index from the lists, in list order. leaves no index if out of memory. */
static void rc_memrefs_index_build(rc_memrefs_t* memrefs) {
  const uint32_t count = rc_memrefs_count_memrefs(memrefs) + rc_memrefs_count_modified_memrefs(memrefs);
  rc_memref_list_t* memref_list;
  rc_modified_memref_list_t* modified_memref_list;
  uint32_t i;

  free(memrefs->index);
  memrefs->index_count = 0;
  memrefs->index_capacity = 64;
  while (count * 4 >= memrefs->index_capacity * 3) /* keep the index at most 3/4 full */
    memrefs->index_capacity <<= 1;

  memrefs->index = (rc_memref_t**)calloc(memrefs->index_capacity, sizeof(rc_memref_t*));
  if (!memrefs->index)
    return;

  for (memref_list = &memrefs->memrefs; memref_list; memref_list = memref_list->next) {
    for (i = 0; i < memref_list->count; ++i)
      rc_memrefs_index_insert(memrefs, &memref_list->items[i]);
  }

  for (modified_memref_list = &memrefs->modified_memrefs; modified_memref_list; modified_memref_list = modified_memref_list->next) {
    for (i = 0; i < modified_memref_list->count; ++i)
      rc_memrefs_index_insert(memrefs, &modified_memref_list->items[i].memref);
  }
}

/* returns the index for a list created by rc_memrefs_init (freed by rc_memrefs_destroy), else NULL */
static rc_memref_t** rc_memrefs_get_index(rc_memrefs_t* memrefs) {
  if (!memrefs->index && memrefs->memrefs.allocated)
    rc_memrefs_index_build(memrefs);

  return memrefs->index;
}

/* adds a new entry (already in its list) to the index, if the list has one */
static void rc_memrefs_index_add(rc_memrefs_t* memrefs, rc_memref_t* memref) {
  if (!memrefs->index)
    return;

  if ((memrefs->index_count + 1) * 4 >= memrefs->index_capacity * 3)
    rc_memrefs_index_build(memrefs);
  else
    rc_memrefs_index_insert(memrefs, memref);
}

rc_memref_t* rc_alloc_memref(rc_parse_state_t* parse, uint32_t address, uint8_t size) {
  rc_memrefs_t* memrefs;
  rc_memref_list_t* memref_list = NULL;
  rc_memref_t* memref = NULL;
  int i;

  for (i = 0; i < 2; i++) {
    if (i == 0) {
      if (!parse->existing_memrefs)
        continue;

      memrefs = parse->existing_memrefs;
    }
    else {
      memrefs = parse->memrefs;
    }

    if (rc_memrefs_get_index(memrefs)) {
      const uint32_t mask = memrefs->index_capacity - 1;
      uint32_t slot = rc_memref_hash(address, size) & mask;

      for (; (memref = memrefs->index[slot]) != NULL; slot = (slot + 1) & mask) {
        if (memref->value.memref_type == RC_MEMREF_TYPE_MEMREF &&
            memref->address == address && memref->value.size == size)
          return memref;
      }

      continue;
    }

    memref_list = &memrefs->memrefs;
    do
    {
      const rc_memref_t* memref_stop;

      memref = memref_list->items;
      memref_stop = memref + memref_list->count;

      for (; memref < memref_stop; ++memref) {
        if (memref->address == address && memref->value.size == size)
          return memref;
      }

      if (!memref_list->next)
        break;

      memref_list = memref_list->next;
    } while (1);
  }

  /* no match found, find a place to put the new entry */
  memref_list = &parse->memrefs->memrefs;
  while (memref_list->count == memref_list->capacity && memref_list->next)
    memref_list = memref_list->next;

  /* create a new entry */
  if (memref_list->count < memref_list->capacity) {
    memref = &memref_list->items[memref_list->count++];
  } else {
    const int32_t old_offset = parse->offset;

    if (memref_list->capacity != 0) {
      memref_list = memref_list->next = RC_ALLOC_SCRATCH(rc_memref_list_t, parse);
      memref_list->next = NULL;
    }

    memref_list->items = RC_ALLOC_ARRAY_SCRATCH(rc_memref_t, 8, parse);
    memref_list->count = 1;
    memref_list->capacity = 8;
    memref_list->allocated = 0;

    memref = memref_list->items;

    /* in preparse mode, don't count this memory, we'll do a single allocation once we have
     * the final total */
    if (!parse->buffer)
      parse->offset = old_offset;
  }

  memset(memref, 0, sizeof(*memref));
  memref->value.memref_type = RC_MEMREF_TYPE_MEMREF;
  memref->value.type = RC_VALUE_TYPE_UNSIGNED;
  memref->value.size = size;
  memref->address = address;

  rc_memrefs_index_add(parse->memrefs, memref);
  return memref;
}

rc_modified_memref_t* rc_alloc_modified_memref(rc_parse_state_t* parse, uint8_t size, const rc_operand_t* parent,
                                               uint8_t modifier_type, const rc_operand_t* modifier) {
  rc_memrefs_t* memrefs;
  rc_modified_memref_list_t* modified_memref_list = NULL;
  rc_modified_memref_t* modified_memref = NULL;
  int i = 0;

  for (i = 0; i < 2; i++) {
    if (i == 0) {
      if (!parse->existing_memrefs)
        continue;

      memrefs = parse->existing_memrefs;
    }
    else {
      memrefs = parse->memrefs;
    }

    if (rc_memrefs_get_index(memrefs)) {
      const uint32_t mask = memrefs->index_capacity - 1;
      uint32_t slot = rc_modified_memref_hash(size, parent, modifier_type, modifier) & mask;
      rc_memref_t* memref;

      for (; (memref = memrefs->index[slot]) != NULL; slot = (slot + 1) & mask) {
        modified_memref = (rc_modified_memref_t*)memref;
        if (memref->value.memref_type == RC_MEMREF_TYPE_MODIFIED_MEMREF &&
            modified_memref->memref.value.size == size &&
            modified_memref->modifier_type == modifier_type &&
            rc_operands_are_equal(&modified_memref->parent, parent) &&
            rc_operands_are_equal(&modified_memref->modifier, modifier)) {
          return modified_memref;
        }
      }

      continue;
    }

    modified_memref_list = &memrefs->modified_memrefs;
    do {
      const rc_modified_memref_t* memref_stop;

      modified_memref = modified_memref_list->items;
      memref_stop = modified_memref + modified_memref_list->count;

      for (; modified_memref < memref_stop; ++modified_memref) {
        if (modified_memref->memref.value.size == size &&
            modified_memref->modifier_type == modifier_type &&
            rc_operands_are_equal(&modified_memref->parent, parent) &&
            rc_operands_are_equal(&modified_memref->modifier, modifier)) {
          return modified_memref;
        }
      }

      if (!modified_memref_list->next)
        break;

      modified_memref_list = modified_memref_list->next;
    } while (1);
  }

  /* no match found, find a place to put the new entry */
  modified_memref_list = &parse->memrefs->modified_memrefs;
  while (modified_memref_list->count == modified_memref_list->capacity && modified_memref_list->next)
    modified_memref_list = modified_memref_list->next;

  /* create a new entry */
  if (modified_memref_list->count < modified_memref_list->capacity) {
    modified_memref = &modified_memref_list->items[modified_memref_list->count++];
  } else {
    const int32_t old_offset = parse->offset;

    if (modified_memref_list->capacity != 0) {
      modified_memref_list = modified_memref_list->next = RC_ALLOC_SCRATCH(rc_modified_memref_list_t, parse);
      modified_memref_list->next = NULL;
    }

    modified_memref_list->items = RC_ALLOC_ARRAY_SCRATCH(rc_modified_memref_t, 8, parse);
    modified_memref_list->count = 1;
    modified_memref_list->capacity = 8;
    modified_memref_list->allocated = 0;

    modified_memref = modified_memref_list->items;

    /* in preparse mode, don't count this memory, we'll do a single allocation once we have
     * the final total */
    if (!parse->buffer)
      parse->offset = old_offset;
  }

  memset(modified_memref, 0, sizeof(*modified_memref));
  modified_memref->memref.value.memref_type = RC_MEMREF_TYPE_MODIFIED_MEMREF;
  modified_memref->memref.value.size = size;
  modified_memref->memref.value.type = rc_memsize_is_float(size) ? RC_VALUE_TYPE_FLOAT : RC_VALUE_TYPE_UNSIGNED;
  memcpy(&modified_memref->parent, parent, sizeof(modified_memref->parent));
  memcpy(&modified_memref->modifier, modifier, sizeof(modified_memref->modifier));
  modified_memref->modifier_type = modifier_type;
  modified_memref->depth = 0;
  modified_memref->memref.address = rc_operand_is_memref(modifier) ? modifier->value.memref->address : modifier->value.num;

  if (rc_operand_is_memref(parent) && parent->value.memref->value.memref_type == RC_MEMREF_TYPE_MODIFIED_MEMREF) {
    const rc_modified_memref_t* parent_modified_memref = (rc_modified_memref_t*)parent->value.memref;
    modified_memref->depth = parent_modified_memref->depth + 1;
  }

  rc_memrefs_index_add(parse->memrefs, &modified_memref->memref);
  return modified_memref;
}

void rc_memrefs_init(rc_memrefs_t* memrefs)
{
  memset(memrefs, 0, sizeof(*memrefs));

  memrefs->memrefs.capacity = 32;
  memrefs->memrefs.items =
    (rc_memref_t*)malloc(memrefs->memrefs.capacity * sizeof(rc_memref_t));
  memrefs->memrefs.allocated = 1;

  memrefs->modified_memrefs.capacity = 16;
  memrefs->modified_memrefs.items =
    (rc_modified_memref_t*)malloc(memrefs->modified_memrefs.capacity * sizeof(rc_modified_memref_t));
  memrefs->modified_memrefs.allocated = 1;
}

void rc_memrefs_destroy(rc_memrefs_t* memrefs)
{
  rc_memref_list_t* memref_list = &memrefs->memrefs;
  rc_modified_memref_list_t* modified_memref_list = &memrefs->modified_memrefs;

  do {
    rc_memref_list_t* current_memref_list = memref_list;
    memref_list = memref_list->next;

    if (current_memref_list->allocated) {
      if (current_memref_list->items)
        free(current_memref_list->items);

      if (current_memref_list != &memrefs->memrefs)
        free(current_memref_list);
    }
  } while (memref_list);

  do {
    rc_modified_memref_list_t* current_modified_memref_list = modified_memref_list;
    modified_memref_list = modified_memref_list->next;

    if (current_modified_memref_list->allocated) {
      if (current_modified_memref_list->items)
        free(current_modified_memref_list->items);

      if (current_modified_memref_list != &memrefs->modified_memrefs)
        free(current_modified_memref_list);
    }
  } while (modified_memref_list);

  if (memrefs->dependencies)
    free(memrefs->dependencies);

  free(memrefs->index);
  free(memrefs);
}

uint32_t rc_memrefs_count_memrefs(const rc_memrefs_t* memrefs)
{
  uint32_t count = 0;
  const rc_memref_list_t* memref_list = &memrefs->memrefs;
  while (memref_list) {
    count += memref_list->count;
    memref_list = memref_list->next;
  }

  return count;
}

uint32_t rc_memrefs_count_modified_memrefs(const rc_memrefs_t* memrefs)
{
  uint32_t count = 0;
  const rc_modified_memref_list_t* modified_memref_list = &memrefs->modified_memrefs;
  while (modified_memref_list) {
    count += modified_memref_list->count;
    modified_memref_list = modified_memref_list->next;
  }

  return count;
}

int rc_parse_memref(const char** memaddr, uint8_t* size, uint32_t* address) {
  const char* aux = *memaddr;
  char* end;
  unsigned long value;

  if (aux[0] == '0') {
    if (aux[1] != 'x' && aux[1] != 'X')
      return RC_INVALID_MEMORY_OPERAND;

    aux += 2;
    switch (*aux++) {
      /* ordered by estimated frequency in case compiler doesn't build a jump table */
      case 'h': case 'H': *size = RC_MEMSIZE_8_BITS; break;
      case ' ':           *size = RC_MEMSIZE_16_BITS; break;
      case 'x': case 'X': *size = RC_MEMSIZE_32_BITS; break;

      case 'm': case 'M': *size = RC_MEMSIZE_BIT_0; break;
      case 'n': case 'N': *size = RC_MEMSIZE_BIT_1; break;
      case 'o': case 'O': *size = RC_MEMSIZE_BIT_2; break;
      case 'p': case 'P': *size = RC_MEMSIZE_BIT_3; break;
      case 'q': case 'Q': *size = RC_MEMSIZE_BIT_4; break;
      case 'r': case 'R': *size = RC_MEMSIZE_BIT_5; break;
      case 's': case 'S': *size = RC_MEMSIZE_BIT_6; break;
      case 't': case 'T': *size = RC_MEMSIZE_BIT_7; break;
      case 'l': case 'L': *size = RC_MEMSIZE_LOW; break;
      case 'u': case 'U': *size = RC_MEMSIZE_HIGH; break;
      case 'k': case 'K': *size = RC_MEMSIZE_BITCOUNT; break;
      case 'w': case 'W': *size = RC_MEMSIZE_24_BITS; break;
      case 'g': case 'G': *size = RC_MEMSIZE_32_BITS_BE; break;
      case 'i': case 'I': *size = RC_MEMSIZE_16_BITS_BE; break;
      case 'j': case 'J': *size = RC_MEMSIZE_24_BITS_BE; break;

      /* case 'v': case 'V': */
      /* case 'y': case 'Y': 64 bit? */
      /* case 'z': case 'Z': 128 bit? */

      case '0':
        if (*aux == 'x') /* user mistyped an extra 0x: 0x0xabcd */
          return RC_INVALID_MEMORY_OPERAND;
        /* fallthrough */

      case '1': case '2': case '3': case '4':
      case '5': case '6': case '7': case '8': case '9':
      case 'a': case 'b': case 'c': case 'd': case 'e': case 'f':
      case 'A': case 'B': case 'C': case 'D': case 'E': case 'F':
        /* legacy support - addresses without a size prefix are assumed to be 16-bit */
        aux--;
        *size = RC_MEMSIZE_16_BITS;
        break;

      default:
        return RC_INVALID_MEMORY_OPERAND;
    }
  }
  else if (aux[0] == 'f' || aux[0] == 'F') {
    ++aux;
    switch (*aux++) {
      case 'f': case 'F': *size = RC_MEMSIZE_FLOAT; break;
      case 'b': case 'B': *size = RC_MEMSIZE_FLOAT_BE; break;
      case 'h': case 'H': *size = RC_MEMSIZE_DOUBLE32; break;
      case 'i': case 'I': *size = RC_MEMSIZE_DOUBLE32_BE; break;
      case 'm': case 'M': *size = RC_MEMSIZE_MBF32; break;
      case 'l': case 'L': *size = RC_MEMSIZE_MBF32_LE; break;

      default:
        return RC_INVALID_FP_OPERAND;
    }
  }
  else {
    return RC_INVALID_MEMORY_OPERAND;
  }

  value = strtoul(aux, &end, 16);

  if (end == aux)
    return RC_INVALID_MEMORY_OPERAND;

  if (value > 0xffffffffU)
    value = 0xffffffffU;

  *address = (uint32_t)value;
  *memaddr = end;
  return RC_OK;
}

static float rc_build_float(uint32_t mantissa_bits, int32_t exponent, int sign) {
  /* 32-bit float has a 23-bit mantissa and 8-bit exponent */
  const uint32_t implied_bit = 1 << 23;
  const uint32_t mantissa = mantissa_bits | implied_bit;
  double dbl = ((double)mantissa) / ((double)implied_bit);

  if (exponent > 127) {
    /* exponent above 127 is a special number */
    if (mantissa_bits == 0) {
      /* infinity */
#ifdef INFINITY /* INFINITY and NAN #defines require C99 */
      dbl = (double)INFINITY;
#else
      dbl = -log(0.0);
#endif
    }
    else {
      /* NaN */
#ifdef NAN
      dbl = NAN;
#else
      dbl = -sqrt(-1);
#endif
    }
  }
  else if (exponent > 0) {
    /* exponent from 1 to 127 is a number greater than 1 */
    while (exponent > 30) {
      dbl *= (double)(1 << 30);
      exponent -= 30;
    }
    dbl *= (double)((long long)1 << exponent);
  }
  else if (exponent < 0) {
    /* exponent from -1 to -127 is a number less than 1 */

    if (exponent == -127) {
      /* exponent -127 (all exponent bits were zero) is a denormalized value
       * (no implied leading bit) with exponent -126 */
      dbl = ((double)mantissa_bits) / ((double)implied_bit);
      exponent = 126;
    } else {
      exponent = -exponent;
    }

    while (exponent > 30) {
      dbl /= (double)(1 << 30);
      exponent -= 30;
    }
    dbl /= (double)((long long)1 << exponent);
  }
  else {
    /* exponent of 0 requires no adjustment */
  }

  return (sign) ? (float)-dbl : (float)dbl;
}

static void rc_transform_memref_float(rc_typed_value_t* value) {
  /* decodes an IEEE 754 float */
  const uint32_t mantissa = (value->value.u32 & 0x7FFFFF);
  const int32_t exponent = (int32_t)((value->value.u32 >> 23) & 0xFF) - 127;
  const int sign = (value->value.u32 & 0x80000000);
  value->value.f32 = rc_build_float(mantissa, exponent, sign);
  value->type = RC_VALUE_TYPE_FLOAT;
}

static void rc_transform_memref_float_be(rc_typed_value_t* value) {
  /* decodes an IEEE 754 float in big endian format */
  const uint32_t mantissa = ((value->value.u32 & 0xFF000000) >> 24) |
                            ((value->value.u32 & 0x00FF0000) >> 8) |
                            ((value->value.u32 & 0x00007F00) << 8);
  const int32_t exponent = (int32_t)(((value->value.u32 & 0x0000007F) << 1) |
                                     ((value->value.u32 & 0x00008000) >> 15)) - 127;
  const int sign = (value->value.u32 & 0x00000080);
  value->value.f32 = rc_build_float(mantissa, exponent, sign);
  value->type = RC_VALUE_TYPE_FLOAT;
}

static void rc_transform_memref_double32(rc_typed_value_t* value)
{
  /* decodes the four most significant bytes of an IEEE 754 double into a float */
  const uint32_t mantissa = (value->value.u32 & 0x000FFFFF) << 3;
  const int32_t exponent = (int32_t)((value->value.u32 >> 20) & 0x7FF) - 1023;
  const int sign = (value->value.u32 & 0x80000000);
  value->value.f32 = rc_build_float(mantissa, exponent, sign);
  value->type = RC_VALUE_TYPE_FLOAT;
}

static void rc_transform_memref_double32_be(rc_typed_value_t* value)
{
  /* decodes the four most significant bytes of an IEEE 754 double in big endian format into a float */
  const uint32_t mantissa = (((value->value.u32 & 0xFF000000) >> 24) |
    ((value->value.u32 & 0x00FF0000) >> 8) |
    ((value->value.u32 & 0x00000F00) << 8)) << 3;
  const int32_t exponent = (int32_t)(((value->value.u32 & 0x0000007F) << 4) |
    ((value->value.u32 & 0x0000F000) >> 12)) - 1023;
  const int sign = (value->value.u32 & 0x00000080);
  value->value.f32 = rc_build_float(mantissa, exponent, sign);
  value->type = RC_VALUE_TYPE_FLOAT;
}

static void rc_transform_memref_mbf32(rc_typed_value_t* value) {
  /* decodes a Microsoft Binary Format float */
  /* NOTE: 32-bit MBF is stored in memory as big endian (at least for Apple II) */
  const uint32_t mantissa = ((value->value.u32 & 0xFF000000) >> 24) |
                            ((value->value.u32 & 0x00FF0000) >> 8) |
                            ((value->value.u32 & 0x00007F00) << 8);
  const int32_t exponent = (int32_t)(value->value.u32 & 0xFF) - 129;
  const int sign = (value->value.u32 & 0x00008000);

  if (mantissa == 0 && exponent == -129)
    value->value.f32 = (sign) ? -0.0f : 0.0f;
  else
    value->value.f32 = rc_build_float(mantissa, exponent, sign);

  value->type = RC_VALUE_TYPE_FLOAT;
}

static void rc_transform_memref_mbf32_le(rc_typed_value_t* value) {
  /* decodes a Microsoft Binary Format float */
  /* Locomotive BASIC (CPC) uses MBF40, but in little endian format */
  const uint32_t mantissa = value->value.u32 & 0x007FFFFF;
  const int32_t exponent = (int32_t)(value->value.u32 >> 24) - 129;
  const int sign = (value->value.u32 & 0x00800000);

  if (mantissa == 0 && exponent == -129)
    value->value.f32 = (sign) ? -0.0f : 0.0f;
  else
    value->value.f32 = rc_build_float(mantissa, exponent, sign);

  value->type = RC_VALUE_TYPE_FLOAT;
}

static const uint8_t rc_bits_set[16] = { 0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4 };

void rc_transform_memref_value(rc_typed_value_t* value, uint8_t size) {
  /* ASSERT: value->type == RC_VALUE_TYPE_UNSIGNED */
  switch (size)
  {
    case RC_MEMSIZE_8_BITS:
      value->value.u32 = (value->value.u32 & 0x000000ff);
      break;

    case RC_MEMSIZE_16_BITS:
      value->value.u32 = (value->value.u32 & 0x0000ffff);
      break;

    case RC_MEMSIZE_24_BITS:
      value->value.u32 = (value->value.u32 & 0x00ffffff);
      break;

    case RC_MEMSIZE_32_BITS:
      break;

    case RC_MEMSIZE_BIT_0:
      value->value.u32 = (value->value.u32 >> 0) & 1;
      break;

    case RC_MEMSIZE_BIT_1:
      value->value.u32 = (value->value.u32 >> 1) & 1;
      break;

    case RC_MEMSIZE_BIT_2:
      value->value.u32 = (value->value.u32 >> 2) & 1;
      break;

    case RC_MEMSIZE_BIT_3:
      value->value.u32 = (value->value.u32 >> 3) & 1;
      break;

    case RC_MEMSIZE_BIT_4:
      value->value.u32 = (value->value.u32 >> 4) & 1;
      break;

    case RC_MEMSIZE_BIT_5:
      value->value.u32 = (value->value.u32 >> 5) & 1;
      break;

    case RC_MEMSIZE_BIT_6:
      value->value.u32 = (value->value.u32 >> 6) & 1;
      break;

    case RC_MEMSIZE_BIT_7:
      value->value.u32 = (value->value.u32 >> 7) & 1;
      break;

    case RC_MEMSIZE_LOW:
      value->value.u32 = value->value.u32 & 0x0f;
      break;

    case RC_MEMSIZE_HIGH:
      value->value.u32 = (value->value.u32 >> 4) & 0x0f;
      break;

    case RC_MEMSIZE_BITCOUNT:
      value->value.u32 = rc_bits_set[(value->value.u32 & 0x0F)]
                       + rc_bits_set[((value->value.u32 >> 4) & 0x0F)];
      break;

    case RC_MEMSIZE_16_BITS_BE:
      value->value.u32 = ((value->value.u32 & 0xFF00) >> 8) |
                         ((value->value.u32 & 0x00FF) << 8);
      break;

    case RC_MEMSIZE_24_BITS_BE:
      value->value.u32 = ((value->value.u32 & 0xFF0000) >> 16) |
                          (value->value.u32 & 0x00FF00) |
                         ((value->value.u32 & 0x0000FF) << 16);
      break;

    case RC_MEMSIZE_32_BITS_BE:
      value->value.u32 = ((value->value.u32 & 0xFF000000) >> 24) |
                         ((value->value.u32 & 0x00FF0000) >> 8) |
                         ((value->value.u32 & 0x0000FF00) << 8) |
                         ((value->value.u32 & 0x000000FF) << 24);
      break;

    case RC_MEMSIZE_FLOAT:
      rc_transform_memref_float(value);
      break;

    case RC_MEMSIZE_FLOAT_BE:
      rc_transform_memref_float_be(value);
      break;

    case RC_MEMSIZE_DOUBLE32:
      rc_transform_memref_double32(value);
      break;

    case RC_MEMSIZE_DOUBLE32_BE:
      rc_transform_memref_double32_be(value);
      break;

    case RC_MEMSIZE_MBF32:
      rc_transform_memref_mbf32(value);
      break;

    case RC_MEMSIZE_MBF32_LE:
      rc_transform_memref_mbf32_le(value);
      break;

    default:
      break;
  }
}

static const uint32_t rc_memref_size_bytes[] = {
  1, /* RC_MEMSIZE_8_BITS     */
  2, /* RC_MEMSIZE_16_BITS    */
  3, /* RC_MEMSIZE_24_BITS    */
  4, /* RC_MEMSIZE_32_BITS    */
  1, /* RC_MEMSIZE_LOW        */
  1, /* RC_MEMSIZE_HIGH       */
  1, /* RC_MEMSIZE_BIT_0      */
  1, /* RC_MEMSIZE_BIT_1      */
  1, /* RC_MEMSIZE_BIT_2      */
  1, /* RC_MEMSIZE_BIT_3      */
  1, /* RC_MEMSIZE_BIT_4      */
  1, /* RC_MEMSIZE_BIT_5      */
  1, /* RC_MEMSIZE_BIT_6      */
  1, /* RC_MEMSIZE_BIT_7      */
  1, /* RC_MEMSIZE_BITCOUNT   */
  2, /* RC_MEMSIZE_16_BITS_BE */
  3, /* RC_MEMSIZE_24_BITS_BE */
  4, /* RC_MEMSIZE_32_BITS_BE */
  4, /* RC_MEMSIZE_FLOAT      */
  4, /* RC_MEMSIZE_MBF32      */
  4, /* RC_MEMSIZE_MBF32_LE   */
  4, /* RC_MEMSIZE_FLOAT_BE   */
  4, /* RC_MEMSIZE_DOUBLE32   */
  4, /* RC_MEMSIZE_DOUBLE32_BE*/
  0, /* RC_MEMSIZE_VARIABLE   */
};

uint32_t rc_memref_bytes(uint8_t size) {
  const size_t index = (size_t)size;
  if (index >= sizeof(rc_memref_size_bytes) / sizeof(rc_memref_size_bytes[0]))
    return 0;

  return rc_memref_size_bytes[index];
}

static const uint32_t rc_memref_masks[] = {
  0x000000ff, /* RC_MEMSIZE_8_BITS     */
  0x0000ffff, /* RC_MEMSIZE_16_BITS    */
  0x00ffffff, /* RC_MEMSIZE_24_BITS    */
  0xffffffff, /* RC_MEMSIZE_32_BITS    */
  0x0000000f, /* RC_MEMSIZE_LOW        */
  0x000000f0, /* RC_MEMSIZE_HIGH       */
  0x00000001, /* RC_MEMSIZE_BIT_0      */
  0x00000002, /* RC_MEMSIZE_BIT_1      */
  0x00000004, /* RC_MEMSIZE_BIT_2      */
  0x00000008, /* RC_MEMSIZE_BIT_3      */
  0x00000010, /* RC_MEMSIZE_BIT_4      */
  0x00000020, /* RC_MEMSIZE_BIT_5      */
  0x00000040, /* RC_MEMSIZE_BIT_6      */
  0x00000080, /* RC_MEMSIZE_BIT_7      */
  0x000000ff, /* RC_MEMSIZE_BITCOUNT   */
  0x0000ffff, /* RC_MEMSIZE_16_BITS_BE */
  0x00ffffff, /* RC_MEMSIZE_24_BITS_BE */
  0xffffffff, /* RC_MEMSIZE_32_BITS_BE */
  0xffffffff, /* RC_MEMSIZE_FLOAT      */
  0xffffffff, /* RC_MEMSIZE_MBF32      */
  0xffffffff, /* RC_MEMSIZE_MBF32_LE   */
  0xffffffff, /* RC_MEMSIZE_FLOAT_BE   */
  0xffffffff, /* RC_MEMSIZE_DOUBLE32   */
  0xffffffff, /* RC_MEMSIZE_DOUBLE32_BE*/
  0xffffffff  /* RC_MEMSIZE_VARIABLE   */
};

uint32_t rc_memref_mask(uint8_t size) {
  const size_t index = (size_t)size;
  if (index >= sizeof(rc_memref_masks) / sizeof(rc_memref_masks[0]))
    return 0xffffffff;

  return rc_memref_masks[index];
}

/* all sizes less than 8-bits (1 byte) are mapped to 8-bits. 24-bit is mapped to 32-bit
 * as we don't expect the client to understand a request for 3 bytes. all other reads are
 * mapped to the little-endian read of the same size. */
static const uint8_t rc_memref_shared_sizes[] = {
  RC_MEMSIZE_8_BITS,  /* RC_MEMSIZE_8_BITS     */
  RC_MEMSIZE_16_BITS, /* RC_MEMSIZE_16_BITS    */
  RC_MEMSIZE_32_BITS, /* RC_MEMSIZE_24_BITS    */
  RC_MEMSIZE_32_BITS, /* RC_MEMSIZE_32_BITS    */
  RC_MEMSIZE_8_BITS,  /* RC_MEMSIZE_LOW        */
  RC_MEMSIZE_8_BITS,  /* RC_MEMSIZE_HIGH       */
  RC_MEMSIZE_8_BITS,  /* RC_MEMSIZE_BIT_0      */
  RC_MEMSIZE_8_BITS,  /* RC_MEMSIZE_BIT_1      */
  RC_MEMSIZE_8_BITS,  /* RC_MEMSIZE_BIT_2      */
  RC_MEMSIZE_8_BITS,  /* RC_MEMSIZE_BIT_3      */
  RC_MEMSIZE_8_BITS,  /* RC_MEMSIZE_BIT_4      */
  RC_MEMSIZE_8_BITS,  /* RC_MEMSIZE_BIT_5      */
  RC_MEMSIZE_8_BITS,  /* RC_MEMSIZE_BIT_6      */
  RC_MEMSIZE_8_BITS,  /* RC_MEMSIZE_BIT_7      */
  RC_MEMSIZE_8_BITS,  /* RC_MEMSIZE_BITCOUNT   */
  RC_MEMSIZE_16_BITS, /* RC_MEMSIZE_16_BITS_BE */
  RC_MEMSIZE_32_BITS, /* RC_MEMSIZE_24_BITS_BE */
  RC_MEMSIZE_32_BITS, /* RC_MEMSIZE_32_BITS_BE */
  RC_MEMSIZE_32_BITS, /* RC_MEMSIZE_FLOAT      */
  RC_MEMSIZE_32_BITS, /* RC_MEMSIZE_MBF32      */
  RC_MEMSIZE_32_BITS, /* RC_MEMSIZE_MBF32_LE   */
  RC_MEMSIZE_32_BITS, /* RC_MEMSIZE_FLOAT_BE   */
  RC_MEMSIZE_32_BITS, /* RC_MEMSIZE_DOUBLE32   */
  RC_MEMSIZE_32_BITS, /* RC_MEMSIZE_DOUBLE32_BE*/
  RC_MEMSIZE_32_BITS  /* RC_MEMSIZE_VARIABLE   */
};

uint8_t rc_memref_shared_size(uint8_t size) {
  const size_t index = (size_t)size;
  if (index >= sizeof(rc_memref_shared_sizes) / sizeof(rc_memref_shared_sizes[0]))
    return size;

  return rc_memref_shared_sizes[index];
}

uint32_t rc_read_memory(uint32_t address, uint8_t size, rc_read_memory_func_t read_memory, void* ud) {
  union buffered_u32 {
    uint32_t u32;
    uint8_t u8[4];
  } value = { 0 };

  uint32_t num_bytes = rc_memref_bytes(size);
  if (num_bytes == 0) /* abort if size is unknown */
    return 0;

  if (!read_memory)
    return 0;

  read_memory(address, value.u8, num_bytes, ud);

  if (rc_is_host_bigendian()) {
    value.u32 = value.u8[0] << 24 |
                value.u8[1] << 16 |
                value.u8[2] << 8 |
                value.u8[3];
  }

  value.u32 &= rc_memref_masks[size]; /* rc_memref_bytes ensures this index exists */
  return value.u32;
}

void rc_update_memref_value(rc_memref_value_t* memref, uint32_t new_value) {
  if (memref->value == new_value) {
    memref->changed = 0;
  }
  else {
    memref->prior = memref->value;
    memref->value = new_value;
    memref->changed = 1;
  }
}

void rc_init_parse_state_memrefs(rc_parse_state_t* parse, rc_memrefs_t* memrefs)
{
  if (memrefs)
    memset(memrefs, 0, sizeof(*memrefs));

  parse->memrefs = memrefs;
}

static uint32_t rc_get_memref_value_value(const rc_memref_value_t* memref, int operand_type) {
  switch (operand_type)
  {
    /* most common case explicitly first, even though it could be handled by default case.
     * this helps the compiler to optimize if it turns the switch into a series of if/elses */
    case RC_OPERAND_ADDRESS:
      return memref->value;

    case RC_OPERAND_DELTA:
      if (!memref->changed) {
        /* fallthrough */
    default:
        return memref->value;
      }
      /* fallthrough */
    case RC_OPERAND_PRIOR:
      return memref->prior;
  }
}

void rc_get_memref_value(rc_typed_value_t* value, rc_memref_t* memref, int operand_type) {
  value->type = memref->value.type;
  value->value.u32 = rc_get_memref_value_value(&memref->value, operand_type);
}

uint32_t rc_get_modified_memref_value(const rc_modified_memref_t* memref, rc_read_memory_func_t read_memory, void* ud) {
  rc_typed_value_t value, modifier;

  rc_evaluate_operand(&value, &memref->parent, NULL);
  rc_evaluate_operand(&modifier, &memref->modifier, NULL);

  switch (memref->modifier_type) {
    case RC_OPERATOR_INDIRECT_READ:
      rc_typed_value_add(&value, &modifier);
      rc_typed_value_convert(&value, RC_VALUE_TYPE_UNSIGNED);
      value.value.u32 = rc_read_memory(value.value.u32, memref->memref.value.size, read_memory, ud);
      value.type = memref->memref.value.type;
      break;

    case RC_OPERATOR_SUB_PARENT:
      /* sub parent is "-parent + modifier" */
      rc_typed_value_negate(&value);
      rc_typed_value_add(&value, &modifier);
      rc_typed_value_convert(&value, memref->memref.value.type);
      break;

    case RC_OPERATOR_SUB_ACCUMULATOR:
      /* sub accumulator is "parent - modifier" */
      rc_typed_value_negate(&modifier);
      /* fallthrough */ /* to case RC_OPERATOR_SUB_ACCUMULATOR */

    case RC_OPERATOR_ADD_ACCUMULATOR:
      /* when modifying the accumulator, force the modifier to match the accumulator
       * type instead of promoting them both to the less restrictive type.
       *
       *   18 - 17.5  will result in an integer. should it be 0 or 1?
       *
       * default: float is less restrictive, convert both to float for combine,
       *          then convert to the memref type.
       *   (int)((float)18 - 17.5) -> (int)(0.5) -> 0
       *
       * accumulator is integer: force modifier to be integer before combining
       *   (int)(18 - (int)17.5) -> (int)(18 - 17) -> 1
       */
      rc_typed_value_convert(&modifier, value.type);
      rc_typed_value_add(&value, &modifier);
      rc_typed_value_convert(&value, memref->memref.value.type);
      break;

    default:
      rc_typed_value_combine(&value, &modifier, memref->modifier_type);
      rc_typed_value_convert(&value, memref->memref.value.type);
      break;
  }

  return value.value.u32;
}

static int rc_operand_is_unchanged(const rc_operand_t* operand) {
  switch (operand->type) {
    case RC_OPERAND_CONST:
    case RC_OPERAND_FP:
    case RC_OPERAND_FUNC:
      return 1;

    case RC_OPERAND_RECALL:
      return 0;

    default:
      /* a variable may be updated more than once a frame. an invalidated memref changes type. */
      if (operand->value.memref->value.memref_type == RC_MEMREF_TYPE_VALUE ||
          operand->value.memref->value.type == RC_VALUE_TYPE_NONE)
        return 0;

      return !operand->value.memref->value.changed;
  }
}

void rc_update_modified_memref_value(rc_modified_memref_t* memref, rc_read_memory_func_t read_memory, void* ud) {
  /* if the inputs did not change this frame or the last, they have the same value, prior and
   * changed flag as when the value was last calculated, so the value would not change. */
  const uint8_t inputs_unchanged = memref->modifier_type != RC_OPERATOR_INDIRECT_READ &&
      rc_operand_is_unchanged(&memref->parent) && rc_operand_is_unchanged(&memref->modifier);

  if (inputs_unchanged && memref->inputs_unchanged)
    memref->memref.value.changed = 0;
  else
    rc_update_memref_value(&memref->memref.value, rc_get_modified_memref_value(memref, read_memory, ud));

  memref->inputs_unchanged = inputs_unchanged;
}

void rc_memrefs_reset_inputs_unchanged(rc_memrefs_t* memrefs) {
  rc_modified_memref_list_t* modified_memref_list = &memrefs->modified_memrefs;
  for (; modified_memref_list; modified_memref_list = modified_memref_list->next) {
    rc_modified_memref_t* modified_memref = modified_memref_list->items;
    const rc_modified_memref_t* modified_memref_stop = modified_memref + modified_memref_list->count;
    for (; modified_memref < modified_memref_stop; ++modified_memref)
      modified_memref->inputs_unchanged = 0;
  }
}

/* Modified memrefs are updated in list order. A modified memref only has to be updated if one of
 * its inputs changed this frame or the last, or if it changed last frame (to clear the changed
 * flag); otherwise rc_update_modified_memref_value would leave it as is. The dependencies index
 * finds them: for each memref and modified memref, the modified memrefs that use it. When a memref
 * changes, its dependents are flagged for an update. A dependent later in the list than a modified
 * memref sees the new value this frame, one earlier in the list sees it next frame. A modified
 * memref that changed, or whose inputs were not unchanged (indirect reads, variables, recalls,
 * invalidated memrefs), is flagged for an update next frame too. */
struct rc_memref_dependencies_t {
  uint32_t num_memrefs;          /* The number of memrefs when the index was built. */
  uint32_t num_modified_memrefs; /* The number of modified memrefs when the index was built. */
  uint32_t* update;              /* One bit per modified memref: update this frame. NULL if not tracked. */
  uint32_t* update_next;         /* One bit per modified memref: update next frame. */
  uint16_t* first_dependent;     /* Per memref, then per modified memref: its first entry in dependents. */
  uint16_t* dependents;          /* Indices of the modified memrefs that use each memref. */
  /* Indirect reads also depend on the memory they read. When the caller says which memory may
   * have changed (rc_memref_dependencies_memory_changed), only the ones that read it are flagged
   * for an update; otherwise every one is. Each is found by where it last read, in a hash of
   * 16-byte pages. */
  uint8_t memory_known;          /* This frame's memory changes were given. */
  uint32_t num_indirect;         /* Indirect reads, in list order. */
  uint32_t* indirect;            /* One bit per modified memref: an indirect read. */
  uint32_t* target;              /* Per indirect read: the address it last read. */
  uint16_t* target_index;        /* Per indirect read: its modified memref index. */
  uint16_t* target_next;         /* Per indirect read: the next one in its bucket. */
  uint8_t* target_bytes;         /* Per indirect read: the bytes it last read. 0: in no bucket. */
  uint16_t* bucket;              /* The first indirect read in each bucket. */
  uint32_t bucket_mask;
};

#define RC_TARGET_PAGE_SHIFT 4
#define RC_TARGET_MAX_BYTES 4
#define RC_TARGET_NONE 0xFFFF

static uint32_t rc_memrefs_find_index(const rc_memrefs_t* memrefs, uint32_t num_memrefs, const rc_memref_t* memref) {
  uint32_t index = 0;

  if (memref->value.memref_type == RC_MEMREF_TYPE_MODIFIED_MEMREF) {
    const rc_modified_memref_t* modified_memref = (const rc_modified_memref_t*)memref;
    const rc_modified_memref_list_t* modified_memref_list = &memrefs->modified_memrefs;
    for (; modified_memref_list; modified_memref_list = modified_memref_list->next) {
      if (modified_memref >= modified_memref_list->items &&
          modified_memref < modified_memref_list->items + modified_memref_list->count)
        return num_memrefs + index + (uint32_t)(modified_memref - modified_memref_list->items);

      index += modified_memref_list->count;
    }
  }
  else {
    const rc_memref_list_t* memref_list = &memrefs->memrefs;
    for (; memref_list; memref_list = memref_list->next) {
      if (memref >= memref_list->items && memref < memref_list->items + memref_list->count)
        return index + (uint32_t)(memref - memref_list->items);

      index += memref_list->count;
    }
  }

  return 0xFFFFFFFF;
}

/* without first_dependent, counts the links. without dependents, counts them per input in
 * first_dependent. otherwise, stores them. returns -1 if an input is not in the lists. */
static int32_t rc_memref_dependencies_link(const rc_memrefs_t* memrefs, uint32_t num_memrefs,
                                           uint16_t* first_dependent, uint16_t* dependents) {
  const rc_modified_memref_list_t* modified_memref_list = &memrefs->modified_memrefs;
  uint32_t index = 0;
  int32_t count = 0;

  for (; modified_memref_list; modified_memref_list = modified_memref_list->next) {
    const rc_modified_memref_t* modified_memref = modified_memref_list->items;
    const rc_modified_memref_t* modified_memref_stop = modified_memref + modified_memref_list->count;

    for (; modified_memref < modified_memref_stop; ++modified_memref, ++index) {
      const rc_operand_t* operand = &modified_memref->parent;
      uint32_t input;

      do {
        /* a modified memref with a variable or recall input is updated every frame, as its inputs
         * are never unchanged */
        if (rc_operand_is_memref(operand) && operand->value.memref->value.memref_type != RC_MEMREF_TYPE_VALUE &&
            (operand == &modified_memref->parent || !rc_operand_is_memref(&modified_memref->parent) ||
             modified_memref->parent.value.memref != operand->value.memref)) {
          input = rc_memrefs_find_index(memrefs, num_memrefs, operand->value.memref);
          if (input == 0xFFFFFFFF)
            return -1;

          if (!first_dependent)
            ++count;
          else if (!dependents)
            ++first_dependent[input];
          else
            dependents[--first_dependent[input]] = (uint16_t)index;
        }

        if (operand == &modified_memref->modifier)
          break;
        operand = &modified_memref->modifier;
      } while (1);
    }
  }

  return count;
}

static rc_memref_dependencies_t* rc_memref_dependencies_create(const rc_memrefs_t* memrefs,
                                                               uint32_t num_memrefs, uint32_t num_modified_memrefs) {
  rc_memref_dependencies_t* dependencies;
  const uint32_t num_inputs = num_memrefs + num_modified_memrefs;
  const uint32_t num_words = (num_modified_memrefs + 31) / 32;
  int32_t num_dependents = (num_inputs <= 0xFFFF) ? rc_memref_dependencies_link(memrefs, num_memrefs, NULL, NULL) : -1;
  const rc_modified_memref_list_t* modified_memref_list;
  uint32_t i, total, num_indirect = 0, num_buckets = 16;

  for (modified_memref_list = &memrefs->modified_memrefs; modified_memref_list; modified_memref_list = modified_memref_list->next)
    for (i = 0; i < modified_memref_list->count; ++i)
      num_indirect += (modified_memref_list->items[i].modifier_type == RC_OPERATOR_INDIRECT_READ);
  while (num_buckets < num_indirect)
    num_buckets <<= 1;

  if (num_dependents < 0 || num_dependents > 0xFFFF) {
    /* can't be tracked: remember that, so every modified memref is updated every frame */
    dependencies = (rc_memref_dependencies_t*)calloc(1, sizeof(*dependencies));
  }
  else {
    dependencies = (rc_memref_dependencies_t*)calloc(1, sizeof(*dependencies) + num_words * 3 * sizeof(uint32_t) +
                                                     (num_inputs + 1 + num_dependents) * sizeof(uint16_t) +
                                                     num_indirect * (sizeof(uint32_t) + 2 * sizeof(uint16_t) + 1) +
                                                     num_buckets * sizeof(uint16_t));
    if (dependencies) {
      dependencies->update = (uint32_t*)(dependencies + 1);
      dependencies->update_next = dependencies->update + num_words;
      dependencies->indirect = dependencies->update_next + num_words;
      dependencies->target = dependencies->indirect + num_words;
      dependencies->first_dependent = (uint16_t*)(dependencies->target + num_indirect);
      dependencies->dependents = dependencies->first_dependent + num_inputs + 1;
      dependencies->target_index = dependencies->dependents + num_dependents;
      dependencies->target_next = dependencies->target_index + num_indirect;
      dependencies->bucket = dependencies->target_next + num_indirect;
      dependencies->target_bytes = (uint8_t*)(dependencies->bucket + num_buckets);
      dependencies->bucket_mask = num_buckets - 1;
      dependencies->num_indirect = num_indirect;
      memset(dependencies->bucket, 0xFF, num_buckets * sizeof(uint16_t));

      for (modified_memref_list = &memrefs->modified_memrefs, i = 0, total = 0; modified_memref_list;
           modified_memref_list = modified_memref_list->next) {
        const rc_modified_memref_t* modified_memref = modified_memref_list->items;
        const rc_modified_memref_t* modified_memref_stop = modified_memref + modified_memref_list->count;
        for (; modified_memref < modified_memref_stop; ++modified_memref, ++i) {
          if (modified_memref->modifier_type == RC_OPERATOR_INDIRECT_READ) {
            dependencies->indirect[i >> 5] |= 1U << (i & 31);
            dependencies->target_index[total++] = (uint16_t)i;
          }
        }
      }

      rc_memref_dependencies_link(memrefs, num_memrefs, dependencies->first_dependent, NULL);
      for (i = 0, total = 0; i <= num_inputs; ++i) {
        total += dependencies->first_dependent[i];
        dependencies->first_dependent[i] = (uint16_t)total;
      }
      rc_memref_dependencies_link(memrefs, num_memrefs, dependencies->first_dependent, dependencies->dependents);

      /* update every modified memref on the first frame */
      for (i = 0; i < num_modified_memrefs; ++i)
        dependencies->update[i >> 5] |= 1U << (i & 31);
    }
  }

  if (dependencies) {
    dependencies->num_memrefs = num_memrefs;
    dependencies->num_modified_memrefs = num_modified_memrefs;
  }

  return dependencies;
}

rc_memref_dependencies_t* rc_memrefs_get_dependencies(rc_memrefs_t* memrefs) {
  rc_memref_dependencies_t* dependencies = memrefs->dependencies;
  uint32_t num_memrefs, num_modified_memrefs;

  /* only memrefs created by rc_memrefs_init are tracked: rc_memrefs_destroy frees the index */
  if (!memrefs->memrefs.allocated)
    return NULL;

  num_modified_memrefs = rc_memrefs_count_modified_memrefs(memrefs);
  if (!num_modified_memrefs)
    return NULL;

  num_memrefs = rc_memrefs_count_memrefs(memrefs);
  if (!dependencies || dependencies->num_memrefs != num_memrefs ||
      dependencies->num_modified_memrefs != num_modified_memrefs) {
    /* memrefs were added (or the state was restored). rebuild the index, which updates
     * every modified memref on the next frame */
    if (dependencies)
      free(dependencies);

    memrefs->dependencies = dependencies = rc_memref_dependencies_create(memrefs, num_memrefs, num_modified_memrefs);
  }

  return (dependencies && dependencies->update) ? dependencies : NULL;
}

void rc_memrefs_update_all_modified_memrefs(rc_memrefs_t* memrefs) {
  /* the index is rebuilt on the next frame, which updates every modified memref */
  if (memrefs->dependencies) {
    free(memrefs->dependencies);
    memrefs->dependencies = NULL;
  }
}

void rc_memref_dependencies_memref_changed(rc_memref_dependencies_t* dependencies, uint32_t memref_index) {
  /* every modified memref is after every memref */
  const uint16_t* dependent = dependencies->dependents + dependencies->first_dependent[memref_index];
  const uint16_t* dependent_stop = dependencies->dependents + dependencies->first_dependent[memref_index + 1];

  for (; dependent < dependent_stop; ++dependent)
    dependencies->update[*dependent >> 5] |= 1U << (*dependent & 31);
}

void rc_memref_dependencies_memory_known(rc_memref_dependencies_t* dependencies) {
  dependencies->memory_known = 1;
}

void rc_memref_dependencies_memory_changed(rc_memref_dependencies_t* dependencies, uint32_t address, uint32_t num_bytes) {
  /* a read of at most RC_TARGET_MAX_BYTES that reaches address starts at most that many bytes - 1 before it */
  const uint32_t last = (address + num_bytes - 1 < address) ? 0xFFFFFFFF : address + num_bytes - 1;
  const uint32_t first_page = (address < RC_TARGET_MAX_BYTES - 1 ? 0 : address - (RC_TARGET_MAX_BYTES - 1)) >> RC_TARGET_PAGE_SHIFT;
  const uint32_t last_page = last >> RC_TARGET_PAGE_SHIFT;
  uint32_t page, i;

  if (!num_bytes || !dependencies->num_indirect)
    return;

  if (last_page - first_page >= dependencies->bucket_mask) {
    /* more pages than buckets: flag every indirect read */
    for (i = 0; i < (dependencies->num_modified_memrefs + 31) / 32; ++i)
      dependencies->update[i] |= dependencies->indirect[i];
    return;
  }

  for (page = first_page; ; ++page) {
    uint32_t k = dependencies->bucket[page & dependencies->bucket_mask];
    for (; k != RC_TARGET_NONE; k = dependencies->target_next[k]) {
      const uint32_t target = dependencies->target[k];
      if (target <= last && (target >= address || address - target < dependencies->target_bytes[k])) {
        const uint32_t index = dependencies->target_index[k];
        dependencies->update[index >> 5] |= 1U << (index & 31);
      }
    }
    if (page == last_page)
      break;
  }
}

typedef struct rc_indirect_read_t {
  rc_read_memory_func_t read_memory;
  void* ud;
  uint32_t address;
  uint32_t num_bytes;
} rc_indirect_read_t;

static uint32_t RC_CCONV rc_indirect_read(uint32_t address, uint8_t* buffer, uint32_t num_bytes, void* ud) {
  rc_indirect_read_t* read = (rc_indirect_read_t*)ud;
  read->address = address;
  read->num_bytes = num_bytes;
  return read->read_memory(address, buffer, num_bytes, read->ud);
}

/* updates the k-th indirect read, and files it under the memory it read. if its inputs did not
 * change, it only has to be updated again when they or that memory change. */
static void rc_update_indirect_read(rc_memref_dependencies_t* dependencies, uint32_t k, rc_modified_memref_t* memref,
                                    rc_read_memory_func_t read_memory, void* ud) {
  rc_indirect_read_t read;
  uint16_t* link;

  read.read_memory = read_memory;
  read.ud = ud;
  read.num_bytes = 0;
  rc_update_modified_memref_value(memref, rc_indirect_read, &read);
  if (read.num_bytes > RC_TARGET_MAX_BYTES)
    read.num_bytes = 0;

  if (dependencies->target_bytes[k] == read.num_bytes && dependencies->target[k] == read.address) {
    /* filed there already */
  }
  else {
    if (dependencies->target_bytes[k]) {
      link = &dependencies->bucket[(dependencies->target[k] >> RC_TARGET_PAGE_SHIFT) & dependencies->bucket_mask];
      while (*link != k)
        link = &dependencies->target_next[*link];
      *link = dependencies->target_next[k];
    }

    dependencies->target[k] = read.address;
    dependencies->target_bytes[k] = (uint8_t)read.num_bytes;
    if (read.num_bytes) {
      link = &dependencies->bucket[(read.address >> RC_TARGET_PAGE_SHIFT) & dependencies->bucket_mask];
      dependencies->target_next[k] = *link;
      *link = (uint16_t)k;
    }
  }

  /* not filed (nothing was read): update it every frame */
  memref->inputs_unchanged = read.num_bytes &&
      rc_operand_is_unchanged(&memref->parent) && rc_operand_is_unchanged(&memref->modifier);
}

static uint32_t rc_lowest_bit_index(uint32_t bits) {
#if defined(__GNUC__)
  return (uint32_t)__builtin_ctz(bits);
#else
  uint32_t index = 0;
  while (!(bits & 1)) {
    bits >>= 1;
    ++index;
  }
  return index;
#endif
}

/* if changed is not NULL, the number of each modified memref that changed (first_number for the
 * first one in the lists, then counting up) is appended to it, in list order */
void rc_update_modified_memref_values(rc_memrefs_t* memrefs, rc_memref_dependencies_t* dependencies,
                                      rc_read_memory_func_t read_memory, void* ud,
                                      uint32_t* changed, uint32_t* num_changed, uint32_t first_number) {
  rc_modified_memref_list_t* modified_memref_list = &memrefs->modified_memrefs;
  uint32_t* update;
  uint32_t* update_next;
  uint32_t first_index = 0, num_words, word, indirect = 0;

  if (!dependencies) {
    if (modified_memref_list->count) {
      do {
        rc_modified_memref_t* modified_memref = modified_memref_list->items;
        const rc_modified_memref_t* modified_memref_stop = modified_memref + modified_memref_list->count;

        for (; modified_memref < modified_memref_stop; ++modified_memref, ++first_number) {
          rc_update_modified_memref_value(modified_memref, read_memory, ud);
          if (changed && modified_memref->memref.value.changed)
            changed[(*num_changed)++] = first_number;
        }

        modified_memref_list = modified_memref_list->next;
      } while (modified_memref_list);
    }
    return;
  }

  update = dependencies->update;
  update_next = dependencies->update_next;
  num_words = (dependencies->num_modified_memrefs + 31) / 32;

  /* without the memory changes, any indirect read may have changed */
  if (!dependencies->memory_known) {
    for (word = 0; word < num_words; ++word)
      update[word] |= dependencies->indirect[word];
  }
  dependencies->memory_known = 0;

  for (word = 0; word < num_words; ++word) {
    uint32_t bits;

    /* dependents flagged while updating are always after the current item, so taking the
     * lowest flagged bit each time visits them in list order */
    while ((bits = update[word]) != 0) {
      const uint32_t index = word * 32 + rc_lowest_bit_index(bits);
      const uint32_t bit = 1U << (index & 31);
      rc_modified_memref_t* modified_memref;
      update[word] = bits & ~bit;

      while (index - first_index >= modified_memref_list->count) {
        first_index += modified_memref_list->count;
        modified_memref_list = modified_memref_list->next;
      }

      modified_memref = &modified_memref_list->items[index - first_index];
      if (dependencies->indirect[word] & bit) {
        /* indirect reads are visited in list order too */
        while (dependencies->target_index[indirect] != index)
          ++indirect;
        rc_update_indirect_read(dependencies, indirect, modified_memref, read_memory, ud);
      }
      else {
        rc_update_modified_memref_value(modified_memref, read_memory, ud);
      }

      if (modified_memref->memref.value.changed) {
        if (changed)
          changed[(*num_changed)++] = first_number + index;

        const uint16_t* dependent = dependencies->dependents +
            dependencies->first_dependent[dependencies->num_memrefs + index];
        const uint16_t* dependent_stop = dependencies->dependents +
            dependencies->first_dependent[dependencies->num_memrefs + index + 1];

        for (; dependent < dependent_stop; ++dependent) {
          if (*dependent > index)
            update[*dependent >> 5] |= 1U << (*dependent & 31);
          else
            update_next[*dependent >> 5] |= 1U << (*dependent & 31);
        }

        /* to clear the changed flag */
        update_next[word] |= bit;
      }
      else if (!modified_memref->inputs_unchanged) {
        update_next[word] |= bit;
      }
    }
  }

  /* update is now empty */
  dependencies->update = update_next;
  dependencies->update_next = update;
}

void rc_update_memref_values(rc_memrefs_t* memrefs, rc_read_memory_func_t read_memory, void* ud) {
  rc_memref_dependencies_t* dependencies = rc_memrefs_get_dependencies(memrefs);
  rc_memref_list_t* memref_list;
  uint32_t index = 0;

  memref_list = &memrefs->memrefs;
  do
  {
    rc_memref_t* memref = memref_list->items;
    const rc_memref_t* memref_stop = memref + memref_list->count;

    for (; memref < memref_stop; ++memref, ++index) {
      if (memref->value.type != RC_VALUE_TYPE_NONE)
        rc_update_memref_value(&memref->value, rc_read_memory(memref->address, memref->value.size, read_memory, ud));

      /* an invalidated memref is never unchanged */
      if (dependencies && (memref->value.changed || memref->value.type == RC_VALUE_TYPE_NONE))
        rc_memref_dependencies_memref_changed(dependencies, index);
    }

    memref_list = memref_list->next;
  } while (memref_list);

  rc_update_modified_memref_values(memrefs, dependencies, read_memory, ud, NULL, NULL, 0);
}
