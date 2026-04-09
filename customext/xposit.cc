// XPosit custom RISC-V extension for Spike
// Opcode: 0x0B (custom-0)
// Posit type: posit<64,2>
// XPosit has its own 32-entry posit register file (separate from F registers).
// Loads/stores move bits between memory and the posit register file.
// Conversions move between posit regs and integer regs (x registers).
// The quire is a hidden per-hart accumulator for QMADD/QMSUB.

// DECODE_MACRO_USAGE_LOGGED must be defined before decode_macros.h is included.
#define DECODE_MACRO_USAGE_LOGGED 0

#include "insn_macros.h"
#include "extension.h"
#include "mmu.h"
#include "decode_macros.h"
#include <array>
#include <cmath>
#include <unordered_map>

// Universal posit library (header-only, included via -I universal/include/sw)
#include <universal/number/posit/posit.hpp>
#include <universal/number/posit/fdp.hpp>   // quire_mul, quire_resolve
#include <universal/number/quire/quire.hpp> // quire<>

// ============================================================================
// Type aliases
// ============================================================================

using P64 = sw::universal::posit<64, 2>;
using Q64 = sw::universal::quire<P64>;   // default capacity

// ============================================================================
// Bit-pattern helpers
// ============================================================================

// Extract raw uint64 bits from posit.
static inline uint64_t p64_to_u64(const P64& p) {
    return static_cast<uint64_t>(p.bits().to_ull());
}

// Construct posit from raw uint64 bit pattern.
static inline P64 u64_to_p64(uint64_t bits) {
    P64 p;
    p.setbits(bits);
    return p;
}

// ============================================================================
// Per-hart posit register file (32 x 64-bit raw posit bits)
// ============================================================================

static std::unordered_map<processor_t*, std::array<uint64_t, 32>> _preg_map;

static inline uint64_t& preg(processor_t* p, int idx) {
    return _preg_map[p][idx];
}

// Read source posit register as a posit value.
#define GET_PS1  P64 ps1 = u64_to_p64(preg(p, insn.rs1()))
#define GET_PS2  P64 ps2 = u64_to_p64(preg(p, insn.rs2()))

// Write result posit to destination posit register.
#define WRITE_PRD(p_val)  preg(p, insn.rd()) = p64_to_u64(p_val)

// ============================================================================
// Quire state — one per hart (processor_t*)
// ============================================================================

static std::unordered_map<processor_t*, Q64> _quire_map;

static inline Q64& get_quire(processor_t* p) {
    return _quire_map[p];
}

// ============================================================================
// Posit negation (two's complement on the raw bit pattern, NaR preserved)
// ============================================================================

static constexpr uint64_t P64_NAR = 0x8000000000000000ULL;

static inline uint64_t posit_negate(uint64_t x) {
    if (x == P64_NAR) return x;  // NaR maps to itself under negation
    return static_cast<uint64_t>(-static_cast<int64_t>(x));
}

// ============================================================================
// Instruction match / mask macros
// ============================================================================

// R-type: [31:27]=funct5, [26:25]=0b10, [24:20]=rs2, [19:15]=rs1,
//         [14:12]=0, [11:7]=rd, [6:0]=0x0B
#define XP_R_MATCH(funct5)  (((uint32_t)(funct5) << 27) | 0x0400000Bu)
#define XP_R_MASK           0xFE00707Fu

// Load (I-type):  [14:12]=funct3, [6:0]=0x0B
#define XP_L_MATCH(funct3)  (((uint32_t)(funct3) << 12) | 0x0Bu)
#define XP_LS_MASK          0x0000707Fu

// Store (S-type): same funct3 field and opcode
#define XP_S_MATCH(funct3)  XP_L_MATCH(funct3)

// ============================================================================
// Load / Store handlers
// ============================================================================

// PLW -- load 32-bit posit from memory into posit register (zero-extend to 64)
static reg_t xp_plw(processor_t* p, insn_t insn, reg_t pc) {
    uint32_t val = MMU.load<uint32_t>(RS1 + insn.i_imm());
    preg(p, insn.rd()) = static_cast<uint64_t>(val);
    return pc + 4;
}

// PLD -- load 64-bit posit from memory into posit register
static reg_t xp_pld(processor_t* p, insn_t insn, reg_t pc) {
    preg(p, insn.rd()) = MMU.load<uint64_t>(RS1 + insn.i_imm());
    return pc + 4;
}

// PSW -- store lower 32 bits of posit register to memory
static reg_t xp_psw(processor_t* p, insn_t insn, reg_t pc) {
    MMU.store<uint32_t>(RS1 + insn.s_imm(), static_cast<uint32_t>(preg(p, insn.rs2())));
    return pc + 4;
}

// PSD -- store 64-bit posit register to memory
static reg_t xp_psd(processor_t* p, insn_t insn, reg_t pc) {
    MMU.store<uint64_t>(RS1 + insn.s_imm(), preg(p, insn.rs2()));
    return pc + 4;
}

// ============================================================================
// Arithmetic instructions
// ============================================================================

// PADD.S
static reg_t xp_padd(processor_t* p, insn_t insn, reg_t pc) {
    GET_PS1; GET_PS2;
    WRITE_PRD(ps1 + ps2);
    return pc + 4;
}

// PSUB.S
static reg_t xp_psub(processor_t* p, insn_t insn, reg_t pc) {
    GET_PS1; GET_PS2;
    WRITE_PRD(ps1 - ps2);
    return pc + 4;
}

// PMUL.S
static reg_t xp_pmul(processor_t* p, insn_t insn, reg_t pc) {
    GET_PS1; GET_PS2;
    WRITE_PRD(ps1 * ps2);
    return pc + 4;
}

// PDIV.S
static reg_t xp_pdiv(processor_t* p, insn_t insn, reg_t pc) {
    GET_PS1; GET_PS2;
    WRITE_PRD(ps1 / ps2);
    return pc + 4;
}

// PMIN.S
static reg_t xp_pmin(processor_t* p, insn_t insn, reg_t pc) {
    GET_PS1; GET_PS2;
    WRITE_PRD(ps1 < ps2 ? ps1 : ps2);
    return pc + 4;
}

// PMAX.S
static reg_t xp_pmax(processor_t* p, insn_t insn, reg_t pc) {
    GET_PS1; GET_PS2;
    WRITE_PRD(ps1 > ps2 ? ps1 : ps2);
    return pc + 4;
}

// PSQRT.S  (rs2 field unused per spec)
static reg_t xp_psqrt(processor_t* p, insn_t insn, reg_t pc) {
    GET_PS1;
    WRITE_PRD(sw::universal::sqrt(ps1));
    return pc + 4;
}

// ============================================================================
// Quire operations
// ============================================================================

// QMADD.S -- quire += ps1 * ps2  (fused, no intermediate rounding)
static reg_t xp_qmadd(processor_t* p, insn_t insn, reg_t pc) {
    GET_PS1; GET_PS2;
    get_quire(p) += sw::universal::quire_mul(ps1, ps2);
    return pc + 4;
}

// QMSUB.S -- quire -= ps1 * ps2
static reg_t xp_qmsub(processor_t* p, insn_t insn, reg_t pc) {
    GET_PS1; GET_PS2;
    get_quire(p) -= sw::universal::quire_mul(ps1, ps2);
    return pc + 4;
}

// QCLR.S -- clear quire to zero
static reg_t xp_qclr(processor_t* p, insn_t insn, reg_t pc) {
    get_quire(p).clear();
    return pc + 4;
}

// QNEG.S -- negate quire
static reg_t xp_qneg(processor_t* p, insn_t insn, reg_t pc) {
    Q64& q = get_quire(p);
    if (!q.iszero())
        q.set_sign(!q.sign());
    return pc + 4;
}

// QROUND.S -- convert quire to posit, write to posit rd
static reg_t xp_qround(processor_t* p, insn_t insn, reg_t pc) {
    P64 result = sw::universal::quire_resolve(get_quire(p));
    WRITE_PRD(result);
    return pc + 4;
}

// ============================================================================
// Conversion instructions (posit <-> integer)
// ============================================================================

// PCVT.W.S -- posit to signed int32, sign-extend to 64
static reg_t xp_pcvt_w_s(processor_t* p, insn_t insn, reg_t pc) {
    GET_PS1;
    WRITE_RD(sext32(static_cast<int32_t>(ps1)));
    return pc + 4;
}

// PCVT.WU.S -- posit to unsigned int32, zero-extend to 64 (RISC-V convention)
static reg_t xp_pcvt_wu_s(processor_t* p, insn_t insn, reg_t pc) {
    GET_PS1;
    // Universal lacks unsigned conversion; use long double to avoid INT64_MAX saturation.
    // Negative posit saturates to 0; value > UINT32_MAX saturates to UINT32_MAX.
    long double ld = static_cast<long double>(ps1);
    uint32_t result;
    if (ld < 0.0L) result = 0;
    else if (ld > 4294967295.0L) result = UINT32_MAX;
    else result = static_cast<uint32_t>(llroundl(ld));
    WRITE_RD(static_cast<reg_t>(result));
    return pc + 4;
}

// PCVT.L.S -- posit to signed int64
static reg_t xp_pcvt_l_s(processor_t* p, insn_t insn, reg_t pc) {
    GET_PS1;
    WRITE_RD(static_cast<int64_t>(ps1));
    return pc + 4;
}

// PCVT.LU.S -- posit to unsigned int64
static reg_t xp_pcvt_lu_s(processor_t* p, insn_t insn, reg_t pc) {
    GET_PS1;
    // Universal lacks unsigned conversion; use long double (80-bit on x86, 64-bit mantissa)
    // to avoid INT64_MAX saturation. Negative posit saturates to 0.
    long double ld = static_cast<long double>(ps1);
    uint64_t result;
    if (ld < 0.0L) result = 0;
    else if (ld >= 18446744073709551616.0L) result = UINT64_MAX;
    else result = static_cast<uint64_t>(ld);
    WRITE_RD(result);
    return pc + 4;
}

// PCVT.S.W -- signed int32 to posit
static reg_t xp_pcvt_s_w(processor_t* p, insn_t insn, reg_t pc) {
    P64 result(static_cast<int32_t>(RS1));
    WRITE_PRD(result);
    return pc + 4;
}

// PCVT.S.WU -- unsigned int32 to posit
static reg_t xp_pcvt_s_wu(processor_t* p, insn_t insn, reg_t pc) {
    P64 result(static_cast<uint32_t>(RS1));
    WRITE_PRD(result);
    return pc + 4;
}

// PCVT.S.L -- signed int64 to posit
static reg_t xp_pcvt_s_l(processor_t* p, insn_t insn, reg_t pc) {
    P64 result(static_cast<int64_t>(RS1));
    WRITE_PRD(result);
    return pc + 4;
}

// PCVT.S.LU -- unsigned int64 to posit
static reg_t xp_pcvt_s_lu(processor_t* p, insn_t insn, reg_t pc) {
    P64 result(static_cast<uint64_t>(RS1));
    WRITE_PRD(result);
    return pc + 4;
}

// ============================================================================
// Sign injection
// Posits use two's complement encoding, so negation is -x (not sign-bit flip).
// PSGNJ.S:  rd = (sign(rs2) == negative) ? negate(rs1) : rs1    [copy sign]
// PSGNJN.S: rd = (sign(rs2) == positive) ? negate(rs1) : rs1    [copy negated sign]
// PSGNJX.S: rd = (sign(rs1) XOR sign(rs2)) ? negate(rs1) : rs1  [xor sign]
//
// Equivalently: if the desired sign differs from rs1's sign, negate rs1.
// sign bit = MSB of the posit raw bits. Positive: MSB=0. Negative: MSB=1.
// ============================================================================

static inline bool posit_sign(uint64_t x) {
    return (x & P64_NAR) != 0;  // MSB set means negative (or NaR)
}

// PSGNJ.S -- rd gets magnitude of rs1 with sign of rs2
static reg_t xp_psgnj(processor_t* p, insn_t insn, reg_t pc) {
    uint64_t a = preg(p, insn.rs1());
    uint64_t b = preg(p, insn.rs2());
    // If rs1 and rs2 have the same sign, keep rs1; otherwise negate rs1.
    uint64_t result = (posit_sign(a) == posit_sign(b)) ? a : posit_negate(a);
    preg(p, insn.rd()) = result;
    return pc + 4;
}

// PSGNJN.S -- rd gets magnitude of rs1 with negated sign of rs2
static reg_t xp_psgnjn(processor_t* p, insn_t insn, reg_t pc) {
    uint64_t a = preg(p, insn.rs1());
    uint64_t b = preg(p, insn.rs2());
    // Desired sign = opposite of rs2's sign.
    // If rs1 already has that sign, keep it; otherwise negate.
    bool desired_negative = !posit_sign(b);
    uint64_t result = (posit_sign(a) == desired_negative) ? a : posit_negate(a);
    preg(p, insn.rd()) = result;
    return pc + 4;
}

// PSGNJX.S -- rd sign = sign(rs1) XOR sign(rs2)
static reg_t xp_psgnjx(processor_t* p, insn_t insn, reg_t pc) {
    uint64_t a = preg(p, insn.rs1());
    uint64_t b = preg(p, insn.rs2());
    bool desired_negative = posit_sign(a) ^ posit_sign(b);
    uint64_t result = (posit_sign(a) == desired_negative) ? a : posit_negate(a);
    preg(p, insn.rd()) = result;
    return pc + 4;
}

// ============================================================================
// Move between posit registers and integer registers (raw bit moves)
// ============================================================================

// PMV.X.W -- copy raw posit register bits to integer register
static reg_t xp_pmv_x_w(processor_t* p, insn_t insn, reg_t pc) {
    WRITE_RD(static_cast<sreg_t>(preg(p, insn.rs1())));
    return pc + 4;
}

// PMV.W.X -- copy integer register bits to posit register
static reg_t xp_pmv_w_x(processor_t* p, insn_t insn, reg_t pc) {
    preg(p, insn.rd()) = RS1;
    return pc + 4;
}

// ============================================================================
// Comparison instructions (write 0/1 to integer register)
// ============================================================================

// PEQ.S
static reg_t xp_peq(processor_t* p, insn_t insn, reg_t pc) {
    GET_PS1; GET_PS2;
    WRITE_RD(ps1 == ps2 ? 1 : 0);
    return pc + 4;
}

// PLT.S
static reg_t xp_plt(processor_t* p, insn_t insn, reg_t pc) {
    GET_PS1; GET_PS2;
    WRITE_RD(ps1 < ps2 ? 1 : 0);
    return pc + 4;
}

// PLE.S
static reg_t xp_ple(processor_t* p, insn_t insn, reg_t pc) {
    GET_PS1; GET_PS2;
    WRITE_RD(ps1 <= ps2 ? 1 : 0);
    return pc + 4;
}

// ============================================================================
// Extension class
// ============================================================================

class xposit_t : public extension_t {
public:
    const char* name() const override { return "xposit"; }

    std::vector<insn_desc_t> get_instructions(const processor_t&) override {
        // Replicate same handler for all 8 slots:
        // fast_rv32i, fast_rv64i, fast_rv32e, fast_rv64e,
        // logged_rv32i, logged_rv64i, logged_rv32e, logged_rv64e
#define INSN8(fn) fn, fn, fn, fn, fn, fn, fn, fn
        return {
            // --- Loads ---
            {XP_L_MATCH(0x1), XP_LS_MASK, INSN8(xp_plw)},  // PLW  funct3=1
            {XP_L_MATCH(0x5), XP_LS_MASK, INSN8(xp_pld)},  // PLD  funct3=5
            // --- Stores ---
            {XP_S_MATCH(0x3), XP_LS_MASK, INSN8(xp_psw)},  // PSW  funct3=3
            {XP_S_MATCH(0x6), XP_LS_MASK, INSN8(xp_psd)},  // PSD  funct3=6
            // --- R-type arithmetic (funct5 in bits [31:27], funct3=0) ---
            {XP_R_MATCH(0x00), XP_R_MASK, INSN8(xp_padd)},
            {XP_R_MATCH(0x01), XP_R_MASK, INSN8(xp_psub)},
            {XP_R_MATCH(0x02), XP_R_MASK, INSN8(xp_pmul)},
            {XP_R_MATCH(0x03), XP_R_MASK, INSN8(xp_pdiv)},
            {XP_R_MATCH(0x04), XP_R_MASK, INSN8(xp_pmin)},
            {XP_R_MATCH(0x05), XP_R_MASK, INSN8(xp_pmax)},
            {XP_R_MATCH(0x06), XP_R_MASK, INSN8(xp_psqrt)},
            // --- Quire ---
            {XP_R_MATCH(0x07), XP_R_MASK, INSN8(xp_qmadd)},
            {XP_R_MATCH(0x08), XP_R_MASK, INSN8(xp_qmsub)},
            {XP_R_MATCH(0x09), XP_R_MASK, INSN8(xp_qclr)},
            {XP_R_MATCH(0x0A), XP_R_MASK, INSN8(xp_qneg)},
            {XP_R_MATCH(0x0B), XP_R_MASK, INSN8(xp_qround)},
            // --- Conversions ---
            {XP_R_MATCH(0x0C), XP_R_MASK, INSN8(xp_pcvt_w_s)},
            {XP_R_MATCH(0x0D), XP_R_MASK, INSN8(xp_pcvt_wu_s)},
            {XP_R_MATCH(0x0E), XP_R_MASK, INSN8(xp_pcvt_l_s)},
            {XP_R_MATCH(0x0F), XP_R_MASK, INSN8(xp_pcvt_lu_s)},
            {XP_R_MATCH(0x10), XP_R_MASK, INSN8(xp_pcvt_s_w)},
            {XP_R_MATCH(0x11), XP_R_MASK, INSN8(xp_pcvt_s_wu)},
            {XP_R_MATCH(0x12), XP_R_MASK, INSN8(xp_pcvt_s_l)},
            {XP_R_MATCH(0x13), XP_R_MASK, INSN8(xp_pcvt_s_lu)},
            // --- Sign injection ---
            {XP_R_MATCH(0x14), XP_R_MASK, INSN8(xp_psgnj)},
            {XP_R_MATCH(0x15), XP_R_MASK, INSN8(xp_psgnjn)},
            {XP_R_MATCH(0x16), XP_R_MASK, INSN8(xp_psgnjx)},
            // --- Raw move ---
            {XP_R_MATCH(0x17), XP_R_MASK, INSN8(xp_pmv_x_w)},
            {XP_R_MATCH(0x18), XP_R_MASK, INSN8(xp_pmv_w_x)},
            // --- Comparisons ---
            {XP_R_MATCH(0x19), XP_R_MASK, INSN8(xp_peq)},
            {XP_R_MATCH(0x1A), XP_R_MASK, INSN8(xp_plt)},
            {XP_R_MATCH(0x1B), XP_R_MASK, INSN8(xp_ple)},
        };
#undef INSN8
    }

    std::vector<disasm_insn_t*> get_disasms(const processor_t*) override {
        return {};  // Traces will show hex encoding; add mnemonics later if needed
    }
};

REGISTER_EXTENSION(xposit, []() { static xposit_t ext; return &ext; })
