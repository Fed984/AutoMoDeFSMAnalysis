library(psych)
error.barsEnhanced <- function (x, stats = NULL, ylab = "Dependent Variable", xlab = "Independent Variable", 
                         main = NULL, eyes = FALSE, ylim = NULL, xlim = NULL, alpha = 0.05, 
                         sd = FALSE, labels = NULL, pos = NULL, arrow.len = 0.05, comparing.lines = FALSE, 
                         comparing.lines.col = "gray50", comparing.lines.lty = 2, comparing.lines.lwd = 0.5,
                         arrow.col = "black", add = FALSE, bars = FALSE, within = FALSE, yaxt="n",
                         col = "blue", ebarrows.lwd = 1, ...) 
{
  SCALE = 0.5
  if (is.null(stats)) {
    x.stats <- describe(x)
    if (within) {
      x.smc <- smc(x, covar = TRUE)
      x.stats$se <- sqrt((x.stats$sd^2 - x.smc)/x.stats$n)
    }
    if (is.null(dim(x))) {
      z <- 1
    }
    else {
      z <- dim(x)[2]
    }
    names <- colnames(x)
  }
  else {
    x.stats <- stats
    z <- dim(x.stats)[1]
    names <- rownames(stats)
  }
  min.x <- min(x.stats$mean, na.rm = TRUE)
  max.x <- max(x.stats$mean, na.rm = TRUE)
  max.se <- max(x.stats$se, na.rm = TRUE)
  {
    if (!sd) {
      if (is.null(stats)) {
        ci <- qt(1 - alpha/2, x.stats$n - 1)
      }
      else {
        ci <- rep(1, z)
      }
    }
    else {
      ci <- sqrt(x.stats$n)
      max.se <- max(ci * x.stats$se, na.rm = TRUE)
    }
  }
  if (is.null(main)) {
    if (!sd) {
      main = paste((1 - alpha) * 100, "% confidence limits", 
                   sep = "")
    }
    else {
      main = paste("Means and standard deviations")
    }
  }
  if (is.null(ylim)) {
    if (is.na(max.x) | is.na(max.se) | is.na(min.x) | is.infinite(max.x) | 
        is.infinite(min.x) | is.infinite(max.se)) {
      ylim = c(0, 1)
    }
    else {
      if (bars) {
        ylim = c(min(0, min.x - 2 * max.se), max.x + 
                   2 * max.se)
      }
      else {
        ylim = c(min.x - 2 * max.se, max.x + 2 * max.se)
      }
    }
  }
  if (bars) {
    mp = barplot(x.stats$mean, ylim = ylim, xlab = xlab, 
                 ylab = ylab, main = main, las = , ...)
    axis(1, mp[1:z], names)
    axis(2)
    box()
  }
  else {
    if (!add) {
      if (missing(xlim)) 
        xlim <- c(0.5, z +  0.5)
      if (is.null(x.stats$values)) {
        plot(x.stats$mean, ylim = ylim, xlab = xlab, 
             ylab = ylab, xlim = xlim, axes = FALSE, main = main, 
             ...)
        axis(1, 1:z, names, ...)
        if(yaxt=="s")
        {
          axis(2)
        }
        box()
      }
      else {
        plot(x.stats$values, x.stats$mean, ylim = ylim, 
             xlab = xlab, ylab = ylab, main = main, ...)
      }
    }
    else {
      points(x.stats$mean, ...)
    }
  }
  if (!is.null(labels)) {
    lab <- labels
  }
  else {
    lab <- paste("V", 1:z, sep = "")
  }
  if (length(pos) == 0) {
    locate <- rep(1, z)
  }
  else {
    locate <- pos
  }
  if (length(labels) == 0) 
    lab <- rep("", z)
  else lab <- labels
  s <- c(1:z)
  if (bars) {
    arrows(mp[s], x.stats$mean[s] - ci[s] * x.stats$se[s], 
           mp[s], x.stats$mean[s] + ci[s] * x.stats$se[s], length = arrow.len, 
           angle = 90, code = 3, col = par("fg"), lty = NULL, 
           lwd = par("lwd"), xpd = NULL)
  }
  else {
    if (is.null(x.stats$values)) {
      arrows(s[s], x.stats$mean[s] - ci[s] * x.stats$se[s], 
             s[s], x.stats$mean[s] + ci[s] * x.stats$se[s], 
             length = arrow.len, angle = 90, code = 3, col = arrow.col,lwd=ebarrows.lwd)
    }
    else {
      arrows(x.stats$values, x.stats$mean[s] - ci[s] * 
               x.stats$se[s], x.stats$values, x.stats$mean[s] +
               ci[s] * x.stats$se[s], length = arrow.len, angle = 90, 
             code = 3, col = arrow.col, lwd = ebarrows.lwd)
    }
    if(comparing.lines)
    {
      abline(h=x.stats$mean[s] - ci[s] * x.stats$se[s], col = comparing.lines.col
             , lty = comparing.lines.lty , lwd = comparing.lines.lwd )
      abline(h=x.stats$mean[s] + ci[s] * x.stats$se[s], col = comparing.lines.col
             , lty = comparing.lines.lty , lwd = comparing.lines.lwd )
    }
    if (eyes) {
      if (length(col) == 1) 
        col <- rep(col, z)
      ln <- seq(-3, 3, 0.1)
      rev <- (length(ln):1)
      for (s in 1:z) {
        if (!is.null(x.stats$n[s])) {
          catseyes(x = s, y = x.stats$mean[s], se = x.stats$se[s], 
                   n = x.stats$n[s], alpha = alpha, density = -10, 
                   col = col[s])
        }
      }
    }
  }
}