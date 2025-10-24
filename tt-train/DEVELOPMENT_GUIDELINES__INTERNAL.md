# Development Guidelines

## Git Commit Workflow

### Branch Management and Push Strategy

**CRITICAL: Preserve work by creating consequent branches**

**Never force-push feature branches:**
- Feature branches serve as backup checkpoints
- Force-pushing destroys history and can lose significant work
- Only force-push the pull request branch after features are stabilized and merged

**Proper branching workflow:**

1. **Initial work on feature branch:**
   ```bash
   git checkout -b ivoitovych/feature-name
   # Make commits
   git push -u myfork ivoitovych/feature-name
   ```

2. **When making changes (amendments, new commits):**
   ```bash
   # Create new consequent branch to preserve backup
   git checkout -b ivoitovych/feature-name-2
   # Make changes/amendments
   git push -u myfork ivoitovych/feature-name-2
   ```

3. **Continue incrementing branch number for each iteration:**
   ```bash
   git checkout -b ivoitovych/feature-name-3
   git push -u myfork ivoitovych/feature-name-3
   ```

**Why this matters:**
- Preserves backup state at each major iteration
- Prevents loss of significant work
- Allows easy rollback to previous states
- Old branches serve as history/reference

**When force-push is acceptable:**
- Only on the final PR branch after feature is stabilized
- Only after confirming the branch is backed up elsewhere
- Never on feature development branches

### Handling Pre-Commit Hooks

**Proper workflow when pre-commit hooks modify files:**

1. Run your commit command with the full message:
   ```bash
   git commit -m "your commit message"
   # or
   git commit --amend -m "your commit message"
   ```

2. Pre-commit hooks (black, clang-format, etc.) will run and may modify files

3. If hooks modify files, the commit will fail with unstaged changes

4. **Review the diff** to see what the hooks changed:
   ```bash
   git diff <modified-file>
   ```

5. Stage the hook-modified files:
   ```bash
   git add <modified-files>
   ```

6. **Run THE SAME commit command again** with THE SAME message:
   ```bash
   git commit -m "your commit message"  # Same command as step 1
   ```

**DO NOT use `git commit --amend --no-edit`** after hooks modify files - this is incorrect workflow.

### Commit Message Policy

**Never mention AI-related content in public commit messages:**
- No references to "Claude", "AI-generated", "LLM", etc.
- Commit messages should appear as normal developer contributions
- This preference applies to ALL commits in public repositories

## Testing Policy

### Comprehensive Testing Requirement

**Policy**: When introducing new features (especially Python bindings, new APIs, or significant functionality), always create comprehensive tests that cover all aspects of the feature.

### Python Bindings Testing

When adding Python bindings for C++ functionality:

1. **Config/Parameter Tests**: Test all configuration parameters can be set and retrieved correctly
   - Test default values
   - Test setting each parameter individually
   - Test setting all parameters together
   - Test parameter types and validation

2. **Object Creation Tests**: Test all ways to create objects
   - Test factory functions (e.g., `create()`)
   - Test constructors
   - Test inheritance (verify base class methods are accessible)

3. **Method Availability Tests**: Test all methods are exposed and callable
   - Test method exists (hasattr)
   - Test method is callable
   - Test basic method invocation

4. **Integration Tests**: Test complete workflows
   - Test end-to-end usage patterns
   - Test parameter access
   - Test state management

5. **Error Handling Tests**: Test error cases
   - Test invalid inputs
   - Test missing files/resources
   - Test type mismatches

6. **Hardware-Dependent Tests**: For tests requiring specific hardware or with stability issues
   - Mark as `@pytest.mark.skip` with clear reason
   - Ensure equivalent functionality is tested in C++
   - Document the skipped tests and their coverage

### Example Test Structure

See `tests/python/test_bert_python_bindings.py` for a comprehensive example:
- 14 config tests
- 7 model creation tests
- 2 weight loading tests
- 2 integration tests
- 3 forward pass tests (skipped, covered by C++)

**Total: 25 passing tests for Python bindings**

### Test Organization

- Group related tests in classes (e.g., `TestBertConfig`, `TestBertModel`)
- Use descriptive test names that explain what is being tested
- Add docstrings explaining the purpose of each test
- Keep tests focused on one aspect each

### Validation

Before committing new bindings or features:

1. Run all related tests: `pytest tests/python/test_<feature>.py -v`
2. Verify test coverage is comprehensive
3. Ensure tests document expected behavior
4. Confirm tests serve as usage examples

## Last Updated

2025-10-23: Initial guidelines based on BERT Python bindings implementation
