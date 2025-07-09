
## Flowline model troubleshooting
### 1. Check for warnings
When you first initialize a new glacier, you may need to modify the bed geometry to remove a positive slope at the top of the glacier. This may also require adjusting the initial thickness profile so the mass flux at the adjusted points is not too high in the first time steps.

### 2. Division by zero (model does not run)
Most likely, all the mass flows out of a grid cell, resulting in division by zero thickness on the next iteration. You may also have somewhere with a slope of zero that is not caught by the warnings.
1. Reduce your timestep.
2. Reduce it again.
3. Play with some temperature values. A glacier should not melt at `T0 = 0`
4. Check your geometry for spots with zero slope (especially if you made modifications).
5. Replace the initial thickness profile with a constant thin layer of ice.
   1. Find the index of the terminus in `h_init`
   2. Use something like `h_init[:145] = 50`, where 145 is the terminus index and 50 is the thickness. Smaller values for thickness may help (you could even try growing from 0).
6. Try a very short simulation (~100 yrs) with a very short timestep (e.g. 0.0125/64 or 0.0125/128).
7. Disable numba on `space_loop()` and use a debugger. Place a breakpoint at `return h, edge` and check `h` and `dhdt` at each timestep.

### 3. Glacier explodes (model runs, but extreme oscillations in thickness profile)
Not enough time for mass flux. Decrease your timestep. You may be able to increase it again once you have a stable glacier profile.

### 4. Weird length oscillations
If your glacier keeps retreating after reaching a specific length, check that it is not the edge of the domain.

### 5. I changed X to get the model to run, but when I change it back the model still crashes?
If you have trouble getting the model to run at all for a tricky glacier but then find something that works, make sure you save the ice profile to use in the next run. Slowly adjusting parameters along with the ice profile is key to difficult initializations.

### 6. The model runs, but the glacier collapses and I have no idea why.
Look at the data in the `model` object. You can see what is available with `dir(model)`. Specifically, try looking at the thickness profile over time with a variable explorer (in Spyder or Pycharm, for example) `pd.DataFrame(model.h)`. Don't forget, your temperature can be too low! Look at where the ELA is for a hint of what direction to go.

### 7. Simulations taking too long?
Initial calibrations using 100 or 500 year simulations can be surprisingly efficient. It does not take long to tell whether the glacier is growing or shrinking.

### Other wacky problems I've had
- Glacer length suddenly went to 0 during spike in cooling for a small glacier with a long timestep. All other metrics stayed normal. Thickness at the top of the glacier went to 0, so the entire glacier was a "detached tongue".
- Small sawtooth steps in length. During large accumulation years, area out in front of the glacier may retain enough ice to exceed the "is glacier" threshold.