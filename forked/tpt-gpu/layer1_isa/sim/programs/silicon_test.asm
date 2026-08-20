;;============================================================================
;; silicon_test.asm — TPT ISA Silicon Verification Test Program
;;============================================================================
;; TPT GPU — Tensor Processing Technology
;; Tests: ALU, immediates, branches, JAL/RET, min/max, byte/halfword ops
;;============================================================================

.org 0x00000000

;; Test 1: Integer ALU
start:
    LW    R1, R0, 4       ; R1 = 100
    LW    R2, R0, 8       ; R2 = 200
    LW    R3, R0, 12      ; R3 = 300

    ADD   R4, R1, R2      ; R4 = 300
    SUB   R5, R3, R1      ; R5 = 200
    MUL   R6, R1, R2      ; R6 = 20000

    AND   R7, R1, R2      ; R7 = 100 & 200 = 64
    OR    R8, R1, R2      ; R8 = 100 | 200 = 236
    XOR   R9, R1, R2      ; R9 = 100 ^ 200 = 172

;; Test 2: Immediate operations
    ADDI  R12, R0, 42     ; R12 = 42
    ADDI  R13, R12, 8     ; R13 = 50
    SUBI  R14, R13, 20    ; R14 = 30

;; Test 3: Store results
    SW    R0, R4, 16      ; data_mem[4] = 300
    SW    R0, R5, 20      ; data_mem[5] = 200
    SW    R0, R6, 24      ; data_mem[6] = 20000
    SW    R0, R14, 28     ; data_mem[7] = 30

;; Test 4: Branch testing
    BEQ   R0, R0, branch_target
    ADDI  R15, R0, 999    ; Skipped

branch_target:
    ADDI  R15, R0, 1      ; R15 = 1

    BNE   R1, R2, bne_ok
    ADDI  R16, R0, 999    ; Skipped

bne_ok:
    ADDI  R16, R0, 2      ; R16 = 2

    BLT   R1, R2, blt_ok
    ADDI  R17, R0, 999    ; Skipped

blt_ok:
    ADDI  R17, R0, 3      ; R17 = 3

;; Test 5: Store branch results
    SW    R0, R15, 32     ; data_mem[8] = 1
    SW    R0, R16, 36     ; data_mem[9] = 2
    SW    R0, R17, 40     ; data_mem[10] = 3

;; Test 6: Jump and link
    JAL   R18, func_add
ret_point:
    SW    R0, R19, 44     ; data_mem[11] = 300

;; Test 7: Min/max
    MIN   R20, R1, R2     ; R20 = 100
    MAX   R21, R1, R2     ; R21 = 200

    SW    R0, R20, 48     ; data_mem[12] = 100
    SW    R0, R21, 52     ; data_mem[13] = 200

;; Final sentinel
    SW    R0, R4, 16      ; data_mem[4] = 300
    SW    R0, R5, 20      ; data_mem[5] = 200
    SW    R0, R14, 28     ; data_mem[7] = 30

done:
    BEQ   R0, R0, done

;; Function: add R1+R2 -> R19
func_add:
    ADD   R19, R1, R2
    RET
