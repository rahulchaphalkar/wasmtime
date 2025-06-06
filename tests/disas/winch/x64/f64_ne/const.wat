;;! target = "x86_64"
;;! test = "winch"

(module
    (func (result i32)
        (f64.const 1.1)
        (f64.const 2.2)
        (f64.ne)
    )
)
;; wasm[0]::function[0]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movq    8(%rdi), %r11
;;       movq    0x10(%r11), %r11
;;       addq    $0x10, %r11
;;       cmpq    %rsp, %r11
;;       ja      0x62
;;   1c: movq    %rdi, %r14
;;       subq    $0x10, %rsp
;;       movq    %rdi, 8(%rsp)
;;       movq    %rsi, (%rsp)
;;       movsd   0x31(%rip), %xmm0
;;       movsd   0x31(%rip), %xmm1
;;       ucomisd %xmm0, %xmm1
;;       movl    $0, %eax
;;       setne   %al
;;       movl    $0, %r11d
;;       setp    %r11b
;;       orq     %r11, %rax
;;       addq    $0x10, %rsp
;;       popq    %rbp
;;       retq
;;   62: ud2
;;   64: addb    %al, (%rax)
;;   66: addb    %al, (%rax)
