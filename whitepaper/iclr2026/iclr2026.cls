% ICLR 2026 Template - LaTeX Class File
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{iclr2026}[2026 ICLR Conference Paper]

% Base document class
\LoadClass{article}

% Packages
\RequirePackage{times}
\RequirePackage{geometry}
\RequirePackage{hyperref}
\RequirePackage{graphicx}
\RequirePackage{url}
\RequirePackage{cite}
\RequirePackage{amsmath,amssymb}
\RequirePackage{booktabs}

% Page geometry
\geometry{
  left=0.9in,
  right=0.9in,
  top=1in,
  bottom=1in,
  includefoot,
  heightrounded
}

% Section formatting
\def\section{\@startsection {section}{1}{\z@}{-3.5ex plus -1ex minus -.2ex}{2.3ex plus .2ex}{\large\bfseries}}
\def\subsection{\@startsection {subsection}{2}{\z@}{-3.25ex plus -1ex minus -.2ex}{1.5ex plus .2ex}{\normalsize\bfseries}}
\def\subsubsection{\@startsection {subsubsection}{3}{\z@}{-3.25ex plus -1ex minus -.2ex}{1.5ex plus .2ex}{\normalsize\itshape}}

% Abstract environment
\renewenvironment{abstract}{%
  \section*{Abstract}%
  \normalsize%
}{}

% Keywords environment
\newenvironment{keywords}{%
  \section*{Keywords}%
  \normalsize%
}{}

% Author formatting
\def\@maketitle{%
  \newpage%
  \null%
  \vskip 2em%
  {\LARGE\@title\par}%
  \vskip 1.5em%
  {\large\@author\par}%
  \vskip 1em%
  \par%
}

% Final measurements
\hbadness=10000
\vbadness=10000

\endinput
