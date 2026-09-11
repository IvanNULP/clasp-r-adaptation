import subprocess
import sys

print("="*70)
print("Перевірка POT (Python Optimal Transport)")
print("="*70)

try:
    import ot
    print(f"\n✓ POT вже встановлено")
    print(f"  Версія: {ot.__version__}")
    print(f"  Шлях: {ot.__file__}")

    required_modules = [
        ("ot.unbalanced", "sinkhorn_unbalanced"),
        ("ot.partial", "partial_wasserstein"),
        ("ot", "dist"),
        ("ot", "emd"),
    ]

    print(f"\n  Перевірка потрібних функцій:")
    for mod_name, func_name in required_modules:
        try:
            mod = __import__(mod_name, fromlist=[func_name])
            func = getattr(mod, func_name)
            print(f"    {mod_name}.{func_name}: ✓ доступна")
        except (ImportError, AttributeError) as e:
            print(f"    {mod_name}.{func_name}: ❌ НЕ доступна ({e})")

except ImportError:
    print("\n❌ POT НЕ встановлено")
    print("\nВстановлюю через pip...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "pot", "--break-system-packages"],
        capture_output=True, text=True
    )
    print(f"\nSTDOUT:\n{result.stdout[-2000:]}")
    if result.returncode != 0:
        print(f"\nSTDERR:\n{result.stderr[-2000:]}")
        print("\n❌ Встановлення не вдалося. Перевір мережевий доступ "
              "(pypi.org має бути в allowed domains).")
    else:
        print("\n✓ Встановлення завершено. Перевіряю імпорт...")
        try:
            import ot
            print(f"✓ POT успішно імпортовано, версія: {ot.__version__}")
        except ImportError as e:
            print(f"❌ Досі не вдається імпортувати: {e}")

print(f"\n{'='*70}")
print("Швидкий функціональний тест (якщо POT доступний)")
print(f"{'='*70}")

try:
    import ot
    import numpy as np

    np.random.seed(42)
    Xs = np.random.randn(50, 3).astype(np.float64)
    Xt = np.random.randn(60, 3).astype(np.float64) + 1.0

    a = np.ones(50) / 50  # рівномірна маса source
    b = np.ones(60) / 60  # рівномірна маса target

    M = ot.dist(Xs, Xt)  # cost matrix (squared euclidean за замовчуванням)
    M /= M.max()

    G0 = ot.emd(a, b, M)
    print(f"  Класичний OT: transport plan shape={G0.shape}, "
          f"total mass={G0.sum():.4f} (очікується ≈1.0)")

    reg = 0.1
    reg_m = 1.0  # marginal relaxation strength
    G_unbalanced = ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg, reg_m)
    print(f"  Unbalanced OT: transport plan shape={G_unbalanced.shape}, "
          f"total mass={G_unbalanced.sum():.4f}")

    s = 0.7  # частка маси для транспортування
    G_partial = ot.partial.partial_wasserstein(a, b, M, m=s)
    print(f"  Partial OT (m={s}): transport plan shape={G_partial.shape}, "
          f"total mass={G_partial.sum():.4f} (очікується ≈{s})")

    print(f"\n✓ Усі три типи OT (classic/unbalanced/partial) працюють коректно")

except ImportError:
    print("  POT недоступний — функціональний тест пропущено")
except Exception as e:
    print(f"  ❌ Помилка під час функціонального тесту: {type(e).__name__}: {e}")