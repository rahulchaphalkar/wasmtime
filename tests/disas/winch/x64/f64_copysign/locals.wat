;;! target = "x86_64"
;;! test = "winch"

(module
    (func (result f64)
        (local $foo f64)  
        (local $bar f64)

        (f64.const -1.1)
        (local.set $foo)

        (f64.const 2.2)
        (local.set $bar)

        (local.get $foo)
        (local.get $bar)
        f64.copysign
    )
)
;; wasm[0]::function[0]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movq    8(%rdi), %r11
;;       movq    0x10(%r11), %r11
;;       addq    $0x20, %r11
;;       cmpq    %rsp, %r11
;;       ja      0x8f
;;   1c: movq    %rdi, %r14
;;       subq    $0x20, %rsp
;;       movq    %rdi, 0x18(%rsp)
;;       movq    %rsi, 0x10(%rsp)
;;       xorq    %r11, %r11
;;       movq    %r11, 8(%rsp)
;;       movq    %r11, (%rsp)
;;       movsd   0x54(%rip), %xmm0
;;       movsd   %xmm0, 8(%rsp)
;;       movsd   0x4e(%rip), %xmm0
;;       movsd   %xmm0, (%rsp)
;;       movsd   (%rsp), %xmm0
;;       movsd   8(%rsp), %xmm1
;;       movabsq $9223372036854775808, %r11
;;       movq    %r11, %xmm15
;;       andpd   %xmm15, %xmm0
;;       andnpd  %xmm1, %xmm15
;;       movaps  %xmm15, %xmm1
;;       orpd    %xmm0, %xmm1
;;       movaps  %xmm1, %xmm0
;;       addq    $0x20, %rsp
;;       popq    %rbp
;;       retq
;;   8f: ud2
;;   91: addb    %al, (%rax)
;;   93: addb    %al, (%rax)
;;   95: addb    %al, (%rax)
;;   97: addb    %bl, -0x66666667(%rdx)
;;   9d: cltd
;;   9e: int1
;;   9f: movl    $0x9999999a, %edi
;;   a4: cltd
;;   a5: cltd
